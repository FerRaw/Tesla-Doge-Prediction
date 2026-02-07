"""
Script de Entrenamiento FINAL - Modelos Mejorados

Entrena solo los modelos que funcionan (boosting optimizado):
- DOGE: XGBoost, LightGBM, CatBoost, Stacking
- TSLA: XGBoost, LightGBM, CatBoost, Stacking
- Impact Classifier: Random Forest, XGBoost, LightGBM

Tiempo estimado: 10-15 minutos

Uso:
    python scripts/02_improved_train_models.py --evaluate --backtesting
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config.settings import settings
from src.data.advanced_features import AdvancedFeatureEngineer
from src.models.improved_predictors import (
    ImprovedDOGEPredictor,
    ImprovedTSLAPredictor,
    ImpactClassifier
)
from src.models.evaluator import (
    ModelEvaluator,
    BacktestEvaluator,
    evaluate_model_complete,
    evaluate_impact_classifier_complete
)


def parse_args():
    """Parsea argumentos"""
    parser = argparse.ArgumentParser(description="Entrena modelos mejorados")
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Ejecuta evaluación completa en test set'
    )
    
    parser.add_argument(
        '--backtesting',
        action='store_true',
        help='Ejecuta simulación de trading'
    )
    
    parser.add_argument(
        '--backtesting-only',
        action='store_true',
        help='Solo ejecuta backtesting (carga modelos existentes)'
    )
    
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Ratio de train/test (default: 0.8)'
    )
    
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Solo evalúa modelos existentes (no entrena)'
    )
    
    return parser.parse_args()


def split_temporal(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split temporal de datos"""
    split_idx = int(len(df) * train_ratio)
    
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    print(f"\n📊 Split de Datos:")
    print(f"   Train: {len(train):,} muestras ({len(train)/len(df):.1%})")
    print(f"   Test:  {len(test):,} muestras ({len(test)/len(df):.1%})")
    print(f"   Train: {train.index.min()} a {train.index.max()}")
    print(f"   Test:  {test.index.min()} a {test.index.max()}")
    
    return train, test


def run_backtesting(predictor, test_df, asset_name: str):
    """Ejecuta backtesting con parámetros realistas"""
    print(f"\n{'='*70}")
    print(f"💰 BACKTESTING {asset_name}")
    print(f"{'='*70}")
    
    target_col = f'TARGET_{asset_name}'
    
    # 1. Obtener predicciones (El modelo ve datos en T)
    raw_predictions = predictor.predict(test_df, model_name='stacking')
    
    # 2. Obtener retornos reales FUTUROS (Lo que pasó en T+1)
    # IMPORTANTE: Hacemos shift(-1) para traer el retorno de la SIGUIENTE vela
    # a la fila actual. Así comparamos:
    # Predicción(T) vs Retorno(T+1)
    future_returns = test_df[target_col].shift(-1).values
    
    # 3. Limpieza de datos
    # El shift(-1) genera un NaN en la última fila (porque no hay futuro después del final)
    # Creamos una máscara para eliminar esos NaNs y alinear todo
    mask = ~np.isnan(future_returns)
    
    # Aplicamos la máscara a ambos arrays para que tengan la misma longitud exacta
    predictions = raw_predictions[mask]
    actual_returns = future_returns[mask]
    
    # Backtesting con parámetros realistas
    backtest_eval = BacktestEvaluator(initial_capital=10000)
    
    print(f"\nProbando diferentes configuraciones de trading...")
    
    # Configuración 1: Conservadora
    print(f"\n📊 Configuración CONSERVADORA:")
    print(f"   - Threshold: 0.5% (solo trades de alta confianza)")
    print(f"   - Position size: 50% del capital")
    print(f"   - Transaction cost: 0.1%")
    
    test_df_recortado = test_df.iloc[mask].copy()
    
    results_conservative = backtest_eval.run_backtest(
        test_df_recortado,
        predictions,
        actual_returns,
        threshold=0.005,
        max_position_size=0.5,
        transaction_cost=0.0005
    )
    backtest_eval.print_backtest_results(results_conservative)
    
    # Configuración 2: Agresiva
    print(f"\n📊 Configuración AGRESIVA:")
    print(f"   - Threshold: 0.1% (más trades)")
    print(f"   - Position size: 100% del capital")
    print(f"   - Transaction cost: 0.1%")
    
    results_aggressive = backtest_eval.run_backtest(
        test_df_recortado,
        predictions,
        actual_returns,
        threshold=0.001,
        max_position_size=1.0,
        transaction_cost=0.0005
    )
    backtest_eval.print_backtest_results(results_aggressive)
    
    # Configuración 3: Moderada
    print(f"\n📊 Configuración MODERADA (RECOMENDADA):")
    print(f"   - Threshold: 0.25% ")
    print(f"   - Position size: 75% del capital")
    print(f"   - Transaction cost: 0.1%")
    
    results_moderate = backtest_eval.run_backtest(
        test_df_recortado,
        predictions,
        actual_returns,
        threshold=0.0025,
        max_position_size=0.75,
        transaction_cost=0.0005
    )
    backtest_eval.print_backtest_results(results_moderate)
    
    return {
        'conservative': results_conservative,
        'aggressive': results_aggressive,
        'moderate': results_moderate
    }


def main():
    """Pipeline de entrenamiento FINAL"""
    args = parse_args()
    
    print("="*70)
    print("🚀 TFM - ENTRENAMIENTO CON MODELOS MEJORADOS")
    print("="*70)
    print(f"Modo: {'EVALUACIÓN ONLY' if args.evaluate_only else 'ENTRENAMIENTO COMPLETO'}")
    print("="*70)
    
    try:
        # ==============================================================
        # PASO 1: Cargar dataset
        # ==============================================================
        print("\n📂 Cargando dataset procesado...")
        
        master_path = settings.DATA_PROCESSED_DIR / settings.FINAL_DATASET_FILE
        
        if not master_path.exists():
            raise FileNotFoundError(
                f"Dataset no encontrado: {master_path}\n"
                f"Ejecuta primero: python scripts/01_preprocess_data.py"
            )
        
        df = pd.read_parquet(master_path)
        print(f"✅ Dataset cargado: {df.shape}")
        
        # ==============================================================
        # PASO 2: Features Avanzadas
        # ==============================================================
        print("\n" + "="*70)
        print("🔬 CREANDO FEATURES AVANZADAS")
        print("="*70)
        
        df_enhanced = AdvancedFeatureEngineer.create_all_advanced_features(df)
        
        print(f"\n✅ Dataset mejorado: {df_enhanced.shape}")
        print(f"   Features originales: {df.shape[1]}")
        print(f"   Features nuevas: {df_enhanced.shape[1] - df.shape[1]}")
        print(f"   Total features: {df_enhanced.shape[1]}")
        
        # ==============================================================
        # PASO 3: Split
        # ==============================================================
        train_df, test_df = split_temporal(df_enhanced, train_ratio=args.split_ratio)
        
        # ==============================================================
        # PASO 4: Entrenar o Cargar modelos
        # ==============================================================
        
        doge_path = settings.MODELS_DIR / "doge_predictor_v2_improved.pkl"
        tsla_path = settings.MODELS_DIR / "tsla_predictor_v2_improved.pkl"
        impact_path = settings.MODELS_DIR / "impact_classifier_v2_improved.pkl"
        
        if args.evaluate_only or args.backtesting_only:
            # Solo evaluar/backtest modelos existentes
            print("\n" + "="*70)
            print("📂 CARGANDO MODELOS EXISTENTES")
            print("="*70)
            
            doge_model = ImprovedDOGEPredictor.load(doge_path)
            tsla_model = ImprovedTSLAPredictor.load(tsla_path)
            
            if not args.backtesting_only:
                impact_model = ImpactClassifier.load(impact_path)
            
        else:
            # Entrenar modelos
            print("\n" + "="*70)
            print("🚀 ENTRENAMIENTO DE MODELOS MEJORADOS")
            print("="*70)
            
            # --- DOGE ---
            print("\n1️⃣ DOGE PREDICTOR")
            doge_model = ImprovedDOGEPredictor(version="v2_improved", use_advanced_models=False)
            doge_model.train(train_df, n_splits=settings.N_CV_SPLITS)
            doge_model.save(doge_path)
            
            # --- TSLA ---
            print("\n2️⃣ TSLA PREDICTOR")
            tsla_model = ImprovedTSLAPredictor(version="v2_improved", use_advanced_models=False)
            tsla_model.train(train_df, n_splits=settings.N_CV_SPLITS)
            tsla_model.save(tsla_path)
            
            # --- IMPACT CLASSIFIER ---
            print("\n3️⃣ IMPACT CLASSIFIER")
            impact_model = ImpactClassifier(version="v2_improved")
            impact_model.train(train_df, n_splits=settings.N_CV_SPLITS)
            impact_model.save(impact_path)
        
        # ==============================================================
        # PASO 5: Evaluación
        # ==============================================================
        if args.evaluate and not args.backtesting_only:
            print("\n" + "="*70)
            print("📊 EVALUACIÓN EN TEST SET")
            print("="*70)
            
            evaluator = ModelEvaluator()
            
            # Evaluar predictores
            evaluate_model_complete(doge_model, test_df, "DOGE", evaluator)
            evaluate_model_complete(tsla_model, test_df, "TSLA", evaluator)
            
            # Evaluar clasificador
            evaluate_impact_classifier_complete(impact_model, test_df, evaluator)
            
            # Comparación
            print("\n" + "="*70)
            print("📋 COMPARACIÓN DE MODELOS DE REGRESIÓN")
            print("="*70)
            comparison = evaluator.compare_models()
            if len(comparison) > 0:
                print(comparison.to_string(index=False))
        
        # ==============================================================
        # PASO 6: Backtesting
        # ==============================================================
        if args.backtesting or args.backtesting_only:
            doge_results = run_backtesting(doge_model, test_df, "DOGE")
            tsla_results = run_backtesting(tsla_model, test_df, "TSLA")
            
            # Guardar resultados de backtesting
            import json
            backtesting_results = {
                'DOGE': {
                    'conservative': {k: v for k, v in doge_results['conservative'].items() if k not in ['equity_curve', 'positions', 'returns_series']},
                    'moderate': {k: v for k, v in doge_results['moderate'].items() if k not in ['equity_curve', 'positions', 'returns_series']},
                    'aggressive': {k: v for k, v in doge_results['aggressive'].items() if k not in ['equity_curve', 'positions', 'returns_series']}
                },
                'TSLA': {
                    'conservative': {k: v for k, v in tsla_results['conservative'].items() if k not in ['equity_curve', 'positions', 'returns_series']},
                    'moderate': {k: v for k, v in tsla_results['moderate'].items() if k not in ['equity_curve', 'positions', 'returns_series']},
                    'aggressive': {k: v for k, v in tsla_results['aggressive'].items() if k not in ['equity_curve', 'positions', 'returns_series']}
                }
            }
            
            results_path = settings.MODELS_DIR / "backtesting_results.json"
            with open(results_path, 'w') as f:
                json.dump(backtesting_results, f, indent=2, default=str)
            
            print(f"\n✅ Resultados de backtesting guardados: {results_path}")
        
        # Si solo backtesting, salir aquí
        if args.backtesting_only:
            print("\n" + "="*70)
            print("✅ BACKTESTING COMPLETADO")
            print("="*70)
            return 0
        
        # ==============================================================
        # RESUMEN FINAL
        # ==============================================================
        print("\n" + "="*70)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*70)
        
        if not args.evaluate_only:
            print(f"\n📁 MODELOS GUARDADOS:")
            print(f"   {doge_path}")
            print(f"   {tsla_path}")
            print(f"   {impact_path}")
        
        print(f"\n📊 MÉTRICAS DE TRAINING (DOGE):")
        for model_name, metrics in doge_model.metrics.items():
            if 'cv_rmse_mean' in metrics:
                print(f"   {model_name:20s}: RMSE = {metrics['cv_rmse_mean']:.6f} (±{metrics['cv_rmse_std']:.6f})")
        
        print(f"\n📊 MÉTRICAS DE TRAINING (TSLA):")
        for model_name, metrics in tsla_model.metrics.items():
            if 'cv_rmse_mean' in metrics:
                print(f"   {model_name:20s}: RMSE = {metrics['cv_rmse_mean']:.6f} (±{metrics['cv_rmse_std']:.6f})")
        
        print(f"\n📊 MÉTRICAS DE TRAINING (IMPACT CLASSIFIER):")
        for model_name, metrics in impact_model.metrics.items():
            if 'accuracy' in metrics:
                print(f"   {model_name:20s}: Accuracy = {metrics['accuracy']:.4f} (±{metrics['std']:.4f})")
        
        print(f"\n🎯 MODELOS INCLUIDOS:")
        print(f"   ✅ XGBoost (mejor directional accuracy)")
        print(f"   ✅ LightGBM (buena generalización)")
        print(f"   ✅ CatBoost (mejor RMSE y R²)")
        print(f"   ✅ Stacking Ensemble (combinación optimizada)")
        
        print(f"\n📈 FEATURES AVANZADAS:")
        print(f"   ✅ Wavelet decomposition")
        print(f"   ✅ Autocorrelación features")
        print(f"   ✅ Interacciones DOGE-TSLA")
        print(f"   ✅ Sentiment interactions")
        print(f"   ✅ Market regime detection")
        
        # Mejor modelo
        best_doge = doge_model.get_best_model_name('cv_rmse_mean', minimize=True)
        best_tsla = tsla_model.get_best_model_name('cv_rmse_mean', minimize=True)
        
        print(f"\n🏆 MEJORES MODELOS (CV):")
        if best_doge:
            print(f"   DOGE: {best_doge.upper()} (RMSE: {doge_model.metrics[best_doge]['cv_rmse_mean']:.6f})")
        if best_tsla:
            print(f"   TSLA: {best_tsla.upper()} (RMSE: {tsla_model.metrics[best_tsla]['cv_rmse_mean']:.6f})")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())