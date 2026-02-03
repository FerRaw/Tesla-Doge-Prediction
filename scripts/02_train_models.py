"""
Script de Entrenamiento Completo
Entrena modelos DOGE, TSLA e Impact Classifier

Uso:
    python scripts/02_train_models.py --evaluate --backtesting
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config.settings import settings
from src.models.predictors import DOGEPredictor, TSLAPredictor, ImpactClassifier
from src.models.evaluator import ModelEvaluator, TradingSimulator


def parse_args():
    """Parsea argumentos"""
    parser = argparse.ArgumentParser(description="Entrena modelos predictivos")
    
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


def evaluate_model(predictor, test_df, asset_name: str, evaluator: ModelEvaluator):
    """Evalúa un modelo en el test set"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUANDO {asset_name}")
    print(f"{'='*70}")
    
    # Preparar datos
    X_test, y_test = predictor.prepare_features(test_df, is_train=True)
    
    # Predicciones con diferentes modelos
    models_to_test = ['xgboost', 'lightgbm', 'elastic_net', 'ensemble']
    
    if 'lstm' in predictor.models:
        models_to_test.insert(-1, 'lstm')  # Antes de ensemble
    
    for model_name in models_to_test:
        pred = predictor.predict(test_df, model_name=model_name)
        evaluator.evaluate_regression(y_test, pred, f"{asset_name}_{model_name}")
        evaluator.print_evaluation(f"{asset_name}_{model_name}")


def run_backtesting(predictor, test_df, asset_name: str):
    """Ejecuta backtesting de trading"""
    print(f"\n{'='*70}")
    print(f"💰 BACKTESTING {asset_name}")
    print(f"{'='*70}")
    
    _, y_test = predictor.prepare_features(test_df, is_train=True)
    pred_ensemble = predictor.predict(test_df, model_name='ensemble')
    
    simulator = TradingSimulator(initial_capital=10000)
    results = simulator.simulate_strategy(pred_ensemble, y_test, threshold=0.01)
    simulator.print_trading_results(results)


def main():
    """Pipeline completo de entrenamiento"""
    args = parse_args()
    
    print("="*70)
    print("🎓 TFM - PIPELINE DE ENTRENAMIENTO")
    print("="*70)
    
    try:
        # ==============================================================
        # PASO 1: Cargar dataset procesado
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
        print(f"   Rango: {df.index.min()} a {df.index.max()}")
        
        # ==============================================================
        # PASO 2: Split temporal
        # ==============================================================
        train_df, test_df = split_temporal(df, train_ratio=args.split_ratio)
        
        # ==============================================================
        # PASO 3: Entrenar modelos
        # ==============================================================
        print("\n" + "="*70)
        print("🚀 ENTRENAMIENTO DE MODELOS")
        print("="*70)
        
        # --- DOGE Predictor ---
        print("\n1️⃣ DOGE PREDICTOR")
        doge_model = DOGEPredictor(version=settings.MODELS_VERSION)
        doge_model.train(train_df, n_splits=settings.N_CV_SPLITS)
        
        # Guardar
        doge_path = settings.MODELS_DIR / f"doge_predictor_{settings.MODELS_VERSION}.pkl"
        doge_model.save(doge_path)
        
        # --- TSLA Predictor ---
        print("\n2️⃣ TSLA PREDICTOR")
        tsla_model = TSLAPredictor(version=settings.MODELS_VERSION)
        tsla_model.train(train_df, n_splits=settings.N_CV_SPLITS)
        
        # Guardar
        tsla_path = settings.MODELS_DIR / f"tsla_predictor_{settings.MODELS_VERSION}.pkl"
        tsla_model.save(tsla_path)
        
        # --- Impact Classifier ---
        print("\n3️⃣ IMPACT CLASSIFIER")
        impact_model = ImpactClassifier(version=settings.MODELS_VERSION)
        impact_model.train(train_df)
        
        # Guardar
        impact_path = settings.MODELS_DIR / f"impact_classifier_{settings.MODELS_VERSION}.pkl"
        impact_model.save(impact_path)
        
        # ==============================================================
        # PASO 4: Evaluación (opcional)
        # ==============================================================
        if args.evaluate:
            print("\n" + "="*70)
            print("📊 EVALUACIÓN EN TEST SET")
            print("="*70)
            
            evaluator = ModelEvaluator()
            
            # Evaluar DOGE
            evaluate_model(doge_model, test_df, "DOGE", evaluator)
            
            # Evaluar TSLA
            evaluate_model(tsla_model, test_df, "TSLA", evaluator)
            
            # Comparación
            print("\n" + "="*70)
            print("📋 COMPARACIÓN DE MODELOS")
            print("="*70)
            comparison = evaluator.compare_models()
            print(comparison[['rmse', 'mae', 'directional_accuracy']].to_string())
        
        # ==============================================================
        # PASO 5: Backtesting (opcional)
        # ==============================================================
        if args.backtesting:
            run_backtesting(doge_model, test_df, "DOGE")
            run_backtesting(tsla_model, test_df, "TSLA")
        
        # ==============================================================
        # RESUMEN FINAL
        # ==============================================================
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        
        print(f"\n📁 MODELOS GUARDADOS:")
        print(f"   {doge_path}")
        print(f"   {tsla_path}")
        print(f"   {impact_path}")
        
        print(f"\n📊 MÉTRICAS DE TRAINING:")
        
        # DOGE metrics
        print(f"\n🐕 DOGE:")
        for model_name, metrics in doge_model.metrics.items():
            if 'cv_rmse_mean' in metrics:
                print(f"   {model_name:15s}: RMSE = {metrics['cv_rmse_mean']:.6f}")
        
        # TSLA metrics
        print(f"\n🚗 TSLA:")
        for model_name, metrics in tsla_model.metrics.items():
            if 'cv_rmse_mean' in metrics:
                print(f"   {model_name:15s}: RMSE = {metrics['cv_rmse_mean']:.6f}")
        
        print(f"\n🚀 PRÓXIMO PASO:")
        print(f"   python scripts/03_run_api.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR EN ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())