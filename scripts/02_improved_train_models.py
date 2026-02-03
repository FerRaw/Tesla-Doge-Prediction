"""
Script de Entrenamiento con Modelos Mejorados
Incluye features avanzadas y arquitecturas potentes

Uso:
    python scripts/02_improved_train_models.py [--full-models] [--evaluate] [--backtesting]
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
from src.models.evaluator import ModelEvaluator, TradingSimulator


def parse_args():
    """Parsea argumentos"""
    parser = argparse.ArgumentParser(description="Entrena modelos mejorados")
    
    parser.add_argument(
        '--full-models',
        action='store_true',
        help='Entrena TODOS los modelos (Transformers, TCN, etc.). Tarda más pero mejor performance.'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Ejecuta evaluación en test set'
    )
    
    parser.add_argument(
        '--backtesting',
        action='store_true',
        help='Ejecuta simulación de trading'
    )
    
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Ratio de train/test (default: 0.8)'
    )
    
    parser.add_argument(
        '--train-impact',
        action='store_true',
        help='Entrena también el clasificador de impacto'
    )
    
    return parser.parse_args()


def split_temporal(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split temporal"""
    split_idx = int(len(df) * train_ratio)
    
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    print(f"\n📊 Split de Datos:")
    print(f"   Train: {len(train):,} muestras ({len(train)/len(df):.1%})")
    print(f"   Test:  {len(test):,} muestras ({len(test)/len(df):.1%})")
    print(f"   Train: {train.index.min()} a {train.index.max()}")
    print(f"   Test:  {test.index.min()} a {test.index.max()}")
    
    return train, test


def evaluate_model(predictor, test_df, asset_name: str, evaluator: ModelEvaluator):
    """Evalúa modelo mejorado"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUANDO {asset_name}")
    print(f"{'='*70}")
    
    X_test, y_test = predictor.prepare_features(test_df, is_train=True)
    
    # Determinar qué modelos probar según el tipo de predictor
    if hasattr(predictor, 'use_advanced_models') and predictor.use_advanced_models:
        # ImprovedDOGEPredictor o ImprovedTSLAPredictor con modelos avanzados
        models_to_test = ['xgboost', 'lightgbm', 'catboost', 'stacking']
        models_to_test.extend(['bilstm_attention', 'tcn', 'transformer', 'cnn_lstm'])
    elif hasattr(predictor, 'use_advanced_models'):
        # ImprovedDOGEPredictor o ImprovedTSLAPredictor sin modelos avanzados
        models_to_test = ['xgboost', 'lightgbm', 'catboost', 'stacking']
    else:
        # TSLAPredictor (versión simple) o DOGEPredictor
        models_to_test = ['xgboost', 'lightgbm', 'elastic_net', 'ensemble']
    
    for model_name in models_to_test:
        if model_name in predictor.models:
            try:
                pred = predictor.predict(test_df, model_name=model_name)
                
                # Ajustar longitud si es modelo secuencial
                if len(pred) != len(y_test):
                    y_test_adj = y_test[-len(pred):]
                else:
                    y_test_adj = y_test
                
                evaluator.evaluate_regression(y_test_adj, pred, f"{asset_name}_{model_name}")
                evaluator.print_evaluation(f"{asset_name}_{model_name}")
            except Exception as e:
                print(f"   ⚠️  Error evaluando {model_name}: {e}")


def evaluate_impact_classifier(classifier, test_df, evaluator: ModelEvaluator):
    """Evalúa clasificador de impacto"""
    print(f"\n{'='*70}")
    print(f"🎯 EVALUANDO IMPACT CLASSIFIER")
    print(f"{'='*70}")
    
    # Modelos a probar
    models_to_test = ['random_forest', 'xgboost', 'lightgbm', 'voting']
    
    for model_name in models_to_test:
        if model_name in classifier.models:
            try:
                # Obtener reporte de clasificación
                report = classifier.get_classification_report(test_df, model_name=model_name)
                
                print(f"\n📊 Modelo: {model_name.upper()}")
                print(f"   Accuracy: {report['accuracy']:.4f}")
                print(f"   Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
                print(f"   Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
                
            except Exception as e:
                print(f"   ⚠️  Error evaluando {model_name}: {e}")


def run_backtesting(predictor, test_df, asset_name: str):
    """Backtesting mejorado"""
    print(f"\n{'='*70}")
    print(f"💰 BACKTESTING {asset_name}")
    print(f"{'='*70}")

    _, y_test = predictor.prepare_features(test_df, is_train=True)

    # Usar el mejor modelo (stacking)
    pred = predictor.predict(test_df, model_name='stacking')

    # Ajustar longitud
    if len(pred) != len(y_test):
        y_test = y_test[-len(pred):]

    simulator = TradingSimulator(initial_capital=10000)
    results = simulator.simulate_strategy(pred, y_test, threshold=0.005)  # Umbral más bajo
    simulator.print_trading_results(results)


def main():
    """Pipeline mejorado"""
    args = parse_args()
    
    print("="*70)
    print("🚀 TFM - ENTRENAMIENTO CON MODELOS MEJORADOS")
    print("="*70)
    print(f"Modo: {'FULL (todos los modelos)' if args.full_models else 'RÁPIDO (solo tree-based + stacking)'}")
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
        # PASO 4: Entrenar modelos
        # ==============================================================
        print("\n" + "="*70)
        print("🚀 ENTRENAMIENTO DE MODELOS MEJORADOS")
        print("="*70)
        
        # --- DOGE ---
        print("\n1️⃣ DOGE PREDICTOR (MEJORADO)")
        doge_model = ImprovedDOGEPredictor(
            version="v2_improved",
            use_advanced_models=args.full_models
        )
        doge_model.train(train_df, n_splits=settings.N_CV_SPLITS)
        
        doge_path = settings.MODELS_DIR / "doge_predictor_v2_improved.pkl"
        doge_model.save(doge_path)
        
        # --- TSLA ---
        print("\n2️⃣ TSLA PREDICTOR (MEJORADO)")
        tsla_model = ImprovedTSLAPredictor(
            version="v2_improved",
            use_advanced_models=args.full_models
        )
        tsla_model.train(train_df, n_splits=settings.N_CV_SPLITS)
        
        tsla_path = settings.MODELS_DIR / "tsla_predictor_v2_improved.pkl"
        tsla_model.save(tsla_path)
        
        
        # --- IMPACT CLASSIFIER (opcional) ---
        impact_model = None
        impact_path = None
        
        print("\n3️⃣ IMPACT CLASSIFIER (MEJORADO)")
        impact_model = ImpactClassifier(version="v2_improved")
        impact_model.train(train_df, n_splits=settings.N_CV_SPLITS)
            
        impact_path = settings.MODELS_DIR / "impact_classifier_v2_improved.pkl"
        impact_model.save(impact_path)
            
        
        # ==============================================================
        # PASO 5: Evaluación
        # ==============================================================
        if args.evaluate:
            print("\n" + "="*70)
            print("📊 EVALUACIÓN EN TEST SET")
            print("="*70)
            
            evaluator = ModelEvaluator()
            
            # doge_model = ImprovedDOGEPredictor(
            # version="v2_improved",
            # use_advanced_models=args.full_models
            # )
            # doge_path = settings.MODELS_DIR / "doge_predictor_v2_improved.pkl"

            # tsla_model = ImprovedTSLAPredictor(
            # version="v2_improved",
            # use_advanced_models=args.full_models
            # )
            # tsla_path = settings.MODELS_DIR / "tsla_predictor_v2_improved.pkl"


            # 3. Cargar los pesos y modelos guardados
            # if doge_path.exists() and tsla_path.exists:
            #     doge_model = doge_model.load(doge_path)
            #     tsla_model = tsla_model.load(tsla_path)
            #     print(f"✅ Modelo DOGE cargado exitosamente desde {doge_path}")
            # else:
            #     raise FileNotFoundError(f"No se encontró el modelo en {doge_path}. ¿Lo has entrenado ya?")

            # Evaluar predictores
            evaluate_model(doge_model, test_df, "DOGE", evaluator)
            evaluate_model(tsla_model, test_df, "TSLA", evaluator)
            
            # Evaluar clasificador si fue entrenado
            if impact_model is not None:
                evaluate_impact_classifier(impact_model, test_df, evaluator)
            
            # Comparación de modelos de regresión
            print("\n" + "="*70)
            print("📋 COMPARACIÓN DE MODELOS DE REGRESIÓN")
            print("="*70)
            comparison = evaluator.compare_models()
            if len(comparison) > 0:
                print(comparison[['rmse', 'mae', 'directional_accuracy']].to_string())
        
        # ==============================================================
        # PASO 6: Backtesting
        # ==============================================================
        if args.backtesting:
            run_backtesting(doge_model, test_df, "DOGE")
            run_backtesting(tsla_model, test_df, "TSLA")
        
        # ==============================================================
        # RESUMEN
        # ==============================================================
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        
        print(f"\n📁 MODELOS GUARDADOS:")
        print(f"   {doge_path}")
        print(f"   {tsla_path}")
        if impact_path:
            print(f"   {impact_path}")
        
        print(f"\n📊 MÉTRICAS DE TRAINING (DOGE):")
        for model_name, metrics in doge_model.metrics.items():
            if 'cv_rmse_mean' in metrics:
                print(f"   {model_name:20s}: RMSE = {metrics['cv_rmse_mean']:.6f} (±{metrics['cv_rmse_std']:.6f})")
        
        print(f"\n📊 MÉTRICAS DE TRAINING (TSLA):")
        for model_name, metrics in tsla_model.metrics.items():
            if 'cv_rmse_mean' in metrics:
                print(f"   {model_name:20s}: RMSE = {metrics['cv_rmse_mean']:.6f} (±{metrics['cv_rmse_std']:.6f})")
        
        if impact_model is not None:
            print(f"\n📊 MÉTRICAS DE TRAINING (IMPACT CLASSIFIER):")
            for model_name, metrics in impact_model.metrics.items():
                if 'cv_accuracy_mean' in metrics:
                    print(f"   {model_name:20s}: Accuracy = {metrics['cv_accuracy_mean']:.4f} (±{metrics['cv_accuracy_std']:.4f})")
        
        print(f"\n🎯 MEJORAS IMPLEMENTADAS:")
        print(f"   ✅ Wavelet decomposition (separación señal/ruido)")
        print(f"   ✅ Autocorrelación features")
        print(f"   ✅ Interacciones DOGE-TSLA (correlación rolling, beta)")
        print(f"   ✅ Sentiment interactions avanzadas")
        print(f"   ✅ Market regime detection")
        print(f"   ✅ XGBoost mejorado (300 trees, regularización)")
        print(f"   ✅ LightGBM mejorado (300 trees, 50 leaves)")
        print(f"   ✅ CatBoost (nuevo, 300 iterations)")
        print(f"   ✅ Stacking Ensemble con meta-learner (GradientBoosting)")
        
        if args.full_models:
            print(f"   ✅ Bi-LSTM con Attention mechanism")
            print(f"   ✅ TCN (Temporal Convolutional Network)")
            print(f"   ✅ Transformer Encoder")
            print(f"   ✅ CNN-LSTM Hybrid")
        
        if args.train_impact:
            print(f"   ✅ Impact Classifier (Random Forest, XGBoost, LightGBM)")
            print(f"   ✅ Voting Ensemble para clasificación")
        
        # Identificar mejor modelo
        best_doge = max(doge_model.metrics.items(), 
                       key=lambda x: -x[1].get('cv_rmse_mean', float('inf')))
        best_tsla = max(tsla_model.metrics.items(), 
                       key=lambda x: -x[1].get('cv_rmse_mean', float('inf')))
        
        print(f"\n🏆 MEJORES MODELOS:")
        print(f"   DOGE: {best_doge[0].upper()} (RMSE: {best_doge[1]['cv_rmse_mean']:.6f})")
        print(f"   TSLA: {best_tsla[0].upper()} (RMSE: {best_tsla[1]['cv_rmse_mean']:.6f})")
        
        if impact_model is not None and len(impact_model.metrics) > 0:
            best_impact = max(impact_model.metrics.items(), 
                            key=lambda x: x[1].get('cv_accuracy_mean', 0))
            print(f"   IMPACT: {best_impact[0].upper()} (Accuracy: {best_impact[1]['cv_accuracy_mean']:.4f})")
        
        print(f"\n🚀 PRÓXIMOS PASOS:")
        print(f"   1. Revisar métricas de validación cruzada")
        if args.evaluate:
            print(f"   2. Analizar resultados en test set")
        else:
            print(f"   2. Ejecutar evaluación: --evaluate")
        
        if args.backtesting:
            print(f"   3. Revisar resultados de backtesting")
        else:
            print(f"   3. Ejecutar backtesting: --backtesting")
        
        print(f"   4. Ejecutar API: python scripts/03_run_api.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())