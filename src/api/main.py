"""
FastAPI Application - TFM Predicción de Mercados con Sentiment Analysis

API RESTful para predicciones de DOGE y TSLA basadas en análisis de sentimiento
de Twitter de Elon Musk.

Autor: Fernando
Versión: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
from pathlib import Path as PathLib
import pandas as pd
import numpy as np
import json
import sys
import tempfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk import bigrams
import re

# Añadir directorio raíz al path
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))

from src.models.improved_predictors import (
    ImprovedDOGEPredictor,
    ImprovedTSLAPredictor,
    ImpactClassifier
)
from src.models.evaluator import ModelEvaluator, BacktestEvaluator
from src.visualization.charts import ChartGenerator
from src.data.advanced_features import AdvancedFeatureEngineer
from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BacktestingRequest, BacktestingResponse, BacktestingMetrics,
    ModelMetrics, ChartResponse, HealthResponse,
    ErrorResponse, EndpointInfo, APIHelp
)

from config.settings import settings
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# =================================================================
# INICIALIZACIÓN DE LA APP
# =================================================================

app = FastAPI(
    title="TFM - Market Prediction API",
    description="API para predicción de mercados (DOGE, TSLA) mediante análisis de sentimiento",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =================================================================
# ESTADO GLOBAL
# =================================================================

class AppState:
    """Estado global de la aplicación"""
    def __init__(self):
        self.doge_model = None
        self.tsla_model = None
        self.impact_model = None
        self.test_df = None
        self.backtesting_results = None
        self.chart_generator = None
        self.models_loaded = False
        self.temp_dir = None  # Directorio temporal para gráficos


state = AppState()

def analyze_ngram_distribution(series_text, extract_func, min_freq, label_name):
   
    all_items = []
    for text in series_text:
        items = extract_func(text, remove_stopwords=True)
        all_items.extend(items)
    
    freq_counter = Counter(all_items)
    
    # Filtrar por frecuencia mínima
    filtered_freq = {k: v for k, v in freq_counter.items() if v >= min_freq}
    top_items = Counter(filtered_freq).most_common(10)
    top_items = [f"{str(item):30s} → {count:5,} veces" for item, count in top_items]
    return filtered_freq, top_items

def extract_words(text, remove_stopwords=True):
    words = text.lower().split()
    
    if remove_stopwords:
        words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return words

def extract_bigrams(text, remove_stopwords=True):
    words = text.lower().split()
    
    if remove_stopwords:
        words = [w for w in words if w not in stop_words and len(w) > 2]
    
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return bigrams

def exploratory_word_analysis(df_clean, min_freq=10):    
    # 1. Analizar Palabras (Unigramas)
    word_freq, top_words = analyze_ngram_distribution(
        df_clean['text_clean'], 
        extract_words, 
        min_freq=min_freq, 
        label_name="🔤 PALABRAS"
    )
    
    # 2. Analizar Bigramas
    bigram_freq, top_bigrams = analyze_ngram_distribution(
        df_clean['text_clean'], 
        extract_bigrams, 
        min_freq=max(min_freq//2, 3), 
        label_name="🔤🔤 BIGRAMAS"
    )
    
    # 3. Empaquetar resultados
    results = {
        'df_clean': df_clean,
        'word_freq': word_freq,
        'bigram_freq': bigram_freq,
        'top_words': top_words,
        'top_bigrams': top_bigrams,
    }
    
    return results

# =================================================================
# EVENTOS DE STARTUP/SHUTDOWN
# =================================================================

@app.on_event("startup")
async def startup_event():
    """Cargar modelos al iniciar la API"""
    print("🚀 Iniciando API...")
    
    try:
        # Cargar modelos
        print("📂 Cargando modelos...")
        doge_path = settings.MODELS_DIR / "doge_predictor_v2_improved.pkl"
        tsla_path = settings.MODELS_DIR / "tsla_predictor_v2_improved.pkl"
        impact_path = settings.MODELS_DIR / "impact_classifier_v2_improved.pkl"
        
        state.doge_model = ImprovedDOGEPredictor.load(doge_path)
        state.tsla_model = ImprovedTSLAPredictor.load(tsla_path)
        state.impact_model = ImpactClassifier.load(impact_path)
        
        # Cargar dataset de test
        print("📂 Cargando dataset...")
        master_path = settings.DATA_PROCESSED_DIR / settings.FINAL_DATASET_FILE
        df = pd.read_parquet(master_path)
        
        # CRÍTICO: Aplicar features avanzadas
        print("🔬 Aplicando features avanzadas...")
        df_enhanced = AdvancedFeatureEngineer.create_all_advanced_features(df)
        
        state.test_df = df_enhanced.iloc[int(len(df_enhanced) * 0.8):]  # 20% final como test
        print(f"✅ Dataset preparado: {state.test_df.shape}")
        
        # Cargar resultados de backtesting
        backtesting_path = settings.MODELS_DIR / "backtesting_results.json"
        if backtesting_path.exists():
            with open(backtesting_path, 'r') as f:
                state.backtesting_results = json.load(f)
        
        # Inicializar generador de gráficos
        state.chart_generator = ChartGenerator()
        
        # Crear directorio temporal para gráficos
        state.temp_dir = PathLib("temp_charts")
        state.temp_dir.mkdir(exist_ok=True)
        
        state.models_loaded = True
        print("✅ API lista!")
        
    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la API"""
    print("👋 Cerrando API...")


# =================================================================
# ENDPOINTS - ROOT & HEALTH
# =================================================================

@app.get("/", tags=["Info"])
async def root():
    """Endpoint raíz - Información de la API"""
    return {
        "name": "TFM - Market Prediction API",
        "version": "1.0.0",
        "status": "running",
        "description": "API para predicción de mercados mediante análisis de sentimiento",
        "docs": "/docs",
        "help": "/help",
        "endpoints": {
            "health": "/health",
            "help": "/help",
            "models": "/models/info",
            "predictions": "/predictions/{asset}",
            "backtesting": "/backtesting/{asset}",
            "charts": "/charts/*"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check - Estado de la API"""
    return HealthResponse(
        status="healthy" if state.models_loaded else "unhealthy",
        version="1.0.0",
        models_loaded={
            "doge": state.doge_model is not None,
            "tsla": state.tsla_model is not None,
            "impact": state.impact_model is not None
        },
        timestamp=datetime.now()
    )


@app.get("/help", response_model=APIHelp, tags=["Info"])
async def api_help():
    """Documentación completa de todos los endpoints"""
    
    endpoints = [
        EndpointInfo(
            path="/",
            method="GET",
            description="Información general de la API",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/health",
            method="GET",
            description="Estado de salud de la API y modelos cargados",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/help",
            method="GET",
            description="Esta ayuda - lista todos los endpoints disponibles",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/analysis/wordcloud",
            method="GET",
            description="WordCloud de palabras más frecuentes de Elon Musk",
            parameters=None,
            example={"path": "/analysis/wordcloud"}
        ),
        EndpointInfo(
            path="/analysis/wordcloud-bigrams",
            method="GET",
            description="WordCloud de bigramas más frecuentes",
            parameters=None,
            example={"path": "/analysis/wordcloud-bigrams"}
        ),
        EndpointInfo(
            path="/analysis/top-words",
            method="GET",
            description="Top N palabras más usadas (gráfico de barras)",
            parameters=["top_n: número de palabras (default 30)"],
            example={"path": "/analysis/top-words?top_n=30"}
        ),
        EndpointInfo(
            path="/analysis/top-bigrams",
            method="GET",
            description="Top N bigramas más usados (gráfico de barras)",
            parameters=["top_n: número de bigramas (default 20)"],
            example={"path": "/analysis/top-bigrams?top_n=20"}
        ),
        EndpointInfo(
            path="/analysis/stats",
            method="GET",
            description="Estadísticas generales del texto de Elon Musk",
            parameters=None,
            example={"path": "/analysis/stats"}
        ),
        EndpointInfo(
            path="/models/info",
            method="GET",
            description="Información detallada de los modelos (métricas, features, etc.)",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/models/performance/{asset}",
            method="GET",
            description="Performance de todos los modelos para un asset",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/models/performance/DOGE"}
        ),
        EndpointInfo(
            path="/predictions/{asset}/latest",
            method="GET",
            description="Última predicción para un asset",
            parameters=["asset: DOGE o TSLA", "model_name: stacking (default)"],
            example={"path": "/predictions/DOGE/latest?model_name=stacking"}
        ),
        EndpointInfo(
            path="/predictions/{asset}/batch",
            method="GET",
            description="Predicciones de los últimos N registros",
            parameters=["asset: DOGE o TSLA", "n: número de predicciones", "model_name"],
            example={"path": "/predictions/DOGE/batch?n=100&model_name=stacking"}
        ),
        EndpointInfo(
            path="/backtesting/{asset}/results",
            method="GET",
            description="Resultados de backtesting pre-computados",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/backtesting/DOGE/results"}
        ),
        EndpointInfo(
            path="/backtesting/{asset}/custom",
            method="POST",
            description="Ejecutar backtesting personalizado en tiempo real",
            parameters=["asset", "threshold", "max_position_size", "transaction_cost"],
            example={
                "body": {
                    "asset": "DOGE",
                    "threshold": 0.0025,
                    "max_position_size": 0.75,
                    "transaction_cost": 0.001,
                    "initial_capital": 10000
                }
            }
        ),
        EndpointInfo(
            path="/charts/predictions/{asset}",
            method="GET",
            description="Gráfico de predicciones vs valores reales",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/charts/predictions/DOGE"}
        ),
        EndpointInfo(
            path="/charts/equity/{asset}",
            method="GET",
            description="Gráfico de equity curve del backtesting",
            parameters=["asset: DOGE o TSLA", "strategy: conservative/moderate/aggressive"],
            example={"path": "/charts/equity/DOGE?strategy=moderate"}
        ),
        EndpointInfo(
            path="/charts/importance/{asset}",
            method="GET",
            description="Gráfico de feature importance",
            parameters=["asset: DOGE o TSLA", "top_n: número de features"],
            example={"path": "/charts/importance/DOGE?top_n=20"}
        ),
        EndpointInfo(
            path="/charts/comparison/{asset}",
            method="GET",
            description="Gráfico comparativo de todos los modelos",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/charts/comparison/DOGE"}
        ),
        EndpointInfo(
            path="/impact/predict",
            method="GET",
            description="Clasificar impacto de los últimos tweets",
            parameters=["n: número de predicciones"],
            example={"path": "/impact/predict?n=10"}
        )
    ]
    
    return APIHelp(
        version="1.0.0",
        description="API para predicción de mercados DOGE y TSLA mediante análisis de sentimiento de Twitter",
        endpoints=endpoints
    )


# =================================================================
# ENDPOINTS - MODELS INFO
# =================================================================

@app.get("/models/info", tags=["Models"])
async def models_info():
    """Información detallada de los modelos"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    return {
        "doge": {
            "name": state.doge_model.model_name,
            "version": state.doge_model.version,
            "models_available": list(state.doge_model.models.keys()),
            "n_features": len(state.doge_model.feature_names) if state.doge_model.feature_names else 0,
            "metrics": state.doge_model.metrics
        },
        "tsla": {
            "name": state.tsla_model.model_name,
            "version": state.tsla_model.version,
            "models_available": list(state.tsla_model.models.keys()),
            "n_features": len(state.tsla_model.feature_names) if state.tsla_model.feature_names else 0,
            "metrics": state.tsla_model.metrics
        },
        "impact": {
            "name": state.impact_model.model_name,
            "version": state.impact_model.version,
            "models_available": list(state.impact_model.models.keys()),
            "metrics": state.impact_model.metrics
        }
    }


@app.get("/models/performance/{asset}", tags=["Models"])
async def model_performance(
    asset: str = Path(..., description="DOGE o TSLA")
):
    """Performance de todos los modelos en test set"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    # Seleccionar modelo
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    # Evaluar cada modelo
    evaluator = ModelEvaluator()
    results = []
    
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'stacking']:
        if model_name in model.models:
            pred = model.predict(state.test_df, model_name=model_name)
            true = state.test_df[target_col].values
            min_len = min(len(pred), len(true))
            
            metrics = evaluator.evaluate_regression(
                true[-min_len:], pred[-min_len:], f"{asset}_{model_name}"
            )
            
            results.append(ModelMetrics(
                model_name=model_name,
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                r2=metrics['r2'],
                directional_accuracy=metrics['directional_accuracy'],
                correlation=metrics['correlation']
            ))
    
    return {
        "asset": asset,
        "models": results
    }


# =================================================================
# ENDPOINTS - PREDICTIONS
# =================================================================

@app.get("/predictions/{asset}/latest", tags=["Predictions"])
async def latest_prediction(
    asset: str = Path(..., description="DOGE o TSLA"),
    model_name: str = Query("stacking", description="xgboost, lightgbm, catboost, stacking")
):
    """Última predicción disponible"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    # Seleccionar modelo
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    
    if model_name not in model.models:
        raise HTTPException(status_code=400, detail=f"Modelo {model_name} no disponible")
    
    # Última predicción
    last_data = state.test_df.tail(1)
    prediction = model.predict(last_data, model_name=model_name)[0]
    
    return PredictionResponse(
        asset=asset,
        model_name=model_name,
        prediction=float(prediction),
        timestamp=datetime.now(),
        confidence=abs(prediction)  # Usar magnitud como "confianza"
    )


@app.get("/predictions/{asset}/batch", tags=["Predictions"])
async def batch_predictions(
    asset: str = Path(..., description="DOGE o TSLA"),
    n: int = Query(100, description="Número de predicciones", ge=1, le=1000),
    model_name: str = Query("stacking", description="Modelo a usar")
):
    """Predicciones de los últimos N registros"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    
    if model_name not in model.models:
        raise HTTPException(status_code=400, detail=f"Modelo {model_name} no disponible")
    
    # Últimas N predicciones
    data = state.test_df.tail(n)
    predictions = model.predict(data, model_name=model_name)
    target_col = f'TARGET_{asset}'
    actuals = data[target_col].values
    
    min_len = min(len(predictions), len(actuals))
    
    return {
        "asset": asset,
        "model_name": model_name,
        "n_predictions": min_len,
        "predictions": predictions[-min_len:].tolist(),
        "actuals": actuals[-min_len:].tolist(),
        "timestamps": data.index[-min_len:].astype(str).tolist()
    }


# =================================================================
# ENDPOINTS - BACKTESTING
# =================================================================

@app.get("/backtesting/{asset}/results", tags=["Backtesting"])
async def backtesting_results(
    asset: str = Path(..., description="DOGE o TSLA")
):
    """Resultados de backtesting pre-computados"""
    if state.backtesting_results is None:
        raise HTTPException(status_code=404, detail="Resultados de backtesting no disponibles")
    
    asset = asset.upper()
    if asset not in state.backtesting_results:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    return {
        "asset": asset,
        "configurations": state.backtesting_results[asset]
    }


@app.post("/backtesting/{asset}/custom", tags=["Backtesting"])
async def custom_backtesting(
    asset: str = Path(..., description="DOGE o TSLA"),
    config: BacktestingRequest = None
):
    """Ejecutar backtesting personalizado"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    # Seleccionar modelo
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    # Predicciones
    predictions = model.predict(state.test_df, model_name='stacking')
    actuals = state.test_df[target_col].values
    min_len = min(len(predictions), len(actuals))
    
    # Backtesting
    backtest_eval = BacktestEvaluator(initial_capital=config.initial_capital)
    results = backtest_eval.run_backtest(
        state.test_df,
        predictions[-min_len:],
        actuals[-min_len:],
        threshold=config.threshold,
        max_position_size=config.max_position_size,
        transaction_cost=config.transaction_cost
    )
    
    metrics = BacktestingMetrics(**{k: v for k, v in results.items() 
                                     if k not in ['equity_curve', 'positions', 'returns_series']})
    
    return BacktestingResponse(
        asset=asset,
        strategy="custom",
        metrics=metrics
    )


# =================================================================
# ENDPOINTS - CHARTS
# =================================================================

@app.get("/charts/predictions/{asset}", tags=["Charts"])
async def chart_predictions(
    asset: str = Path(..., description="DOGE o TSLA"),
    model_name: str = Query("stacking", description="Modelo a usar")
):
    """
    Gráfico de predicciones vs valores reales
    
    Devuelve un archivo PNG para descargar
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    pred = model.predict(state.test_df, model_name=model_name)
    true = state.test_df[target_col].values
    min_len = min(len(pred), len(true))
    
    # Crear gráfico y guardar
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot
    axes[0, 0].scatter(true[-min_len:], pred[-min_len:], alpha=0.5, s=20)
    axes[0, 0].plot([true[-min_len:].min(), true[-min_len:].max()], 
                    [true[-min_len:].min(), true[-min_len:].max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Returns')
    axes[0, 0].set_ylabel('Predicted Returns')
    axes[0, 0].set_title(f'{asset} - Predictions vs Actual ({model_name})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series
    indices = np.arange(min_len)
    axes[0, 1].plot(indices, true[-min_len:], label='Actual', alpha=0.7, linewidth=1)
    axes[0, 1].plot(indices, pred[-min_len:], label='Predicted', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Time Index')
    axes[0, 1].set_ylabel('Returns')
    axes[0, 1].set_title(f'{asset} - Time Series Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors = pred[-min_len:] - true[-min_len:]
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{asset} - Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Directional accuracy
    window = 50
    direction_true = np.sign(true[-min_len:])
    direction_pred = np.sign(pred[-min_len:])
    correct = (direction_true == direction_pred).astype(int)
    rolling_acc = pd.Series(correct).rolling(window).mean() * 100
    
    axes[1, 1].plot(rolling_acc, linewidth=2)
    axes[1, 1].axhline(y=50, color='r', linestyle='--', label='Random (50%)')
    axes[1, 1].set_xlabel('Time Index')
    axes[1, 1].set_ylabel('Directional Accuracy (%)')
    axes[1, 1].set_title(f'{asset} - Rolling Directional Accuracy (window={window})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Guardar archivo
    filepath = state.temp_dir / f"predictions_{asset}_{model_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"predictions_{asset}_{model_name}.png"
    )


@app.get("/charts/equity/{asset}", tags=["Charts"])
async def chart_equity(
    asset: str = Path(..., description="DOGE o TSLA"),
    strategy: str = Query("moderate", description="conservative, moderate, aggressive")
):
    """
    Gráfico de equity curve del backtesting
    
    Devuelve un archivo PNG para descargar
    """
    if state.backtesting_results is None:
        raise HTTPException(status_code=404, detail="Resultados de backtesting no disponibles")
    
    asset = asset.upper()
    if strategy not in ['conservative', 'moderate', 'aggressive']:
        raise HTTPException(status_code=400, detail="Strategy debe ser conservative, moderate o aggressive")
    
    # RE-EJECUTAR backtesting para obtener equity_curve
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    try:
        predictions = model.predict(state.test_df, model_name='stacking')
        actuals = state.test_df[target_col].values
        min_len = min(len(predictions), len(actuals))
        
        # Configuraciones
        configs = {
            'conservative': {'threshold': 0.005, 'position': 0.5, 'cost': 0.001},
            'moderate': {'threshold': 0.0025, 'position': 0.75, 'cost': 0.001},
            'aggressive': {'threshold': 0.001, 'position': 1.0, 'cost': 0.001}
        }
        
        config = configs[strategy]
        backtest_eval = BacktestEvaluator(initial_capital=10000)
        results = backtest_eval.run_backtest(
            state.test_df,
            predictions[-min_len:],
            actuals[-min_len:],
            threshold=config['threshold'],
            max_position_size=config['position'],
            transaction_cost=config['cost']
        )
        
        equity = np.array(results['equity_curve'])
        
        # Crear gráfico
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. Equity curve
        axes[0].plot(equity, linewidth=2, color='#2E86AB')
        axes[0].fill_between(range(len(equity)), equity, alpha=0.3, color='#2E86AB')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title(f'{asset} - Equity Curve ({strategy.capitalize()})')
        axes[0].grid(True, alpha=0.3)
        
        # Estadísticas
        final_value = equity[-1]
        max_value = equity.max()
        initial_value = equity[0]
        
        axes[0].axhline(y=initial_value, color='gray', linestyle='--', 
                       label=f'Initial: ${initial_value:,.0f}')
        axes[0].axhline(y=max_value, color='green', linestyle='--', 
                       label=f'Peak: ${max_value:,.0f}')
        axes[0].legend()
        
        # 2. Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, 
                            color='red', alpha=0.3)
        axes[1].plot(drawdown, color='red', linewidth=1)
        axes[1].set_xlabel('Time Index')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_title(f'{asset} - Drawdown Analysis')
        axes[1].grid(True, alpha=0.3)
        
        max_dd = drawdown.min()
        axes[1].axhline(y=max_dd, color='darkred', linestyle='--',
                       label=f'Max DD: {max_dd:.2f}%')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Guardar archivo
        filepath = state.temp_dir / f"equity_{asset}_{strategy}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return FileResponse(
            filepath,
            media_type="image/png",
            filename=f"equity_{asset}_{strategy}.png"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")


@app.get("/charts/importance/{asset}", tags=["Charts"])
async def chart_importance(
    asset: str = Path(..., description="DOGE o TSLA"),
    top_n: int = Query(20, description="Número de features", ge=5, le=50)
):
    """Gráfico de feature importance - Descarga PNG"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    
    # Usar mejor modelo según asset
    model_name = 'catboost' if asset == 'DOGE' else 'xgboost'
    importance = model.get_feature_importance(model_name, top_n=top_n)
    
    # Crear gráfico
    features = list(importance.keys())
    importances = list(importance.values())
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.4)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors)
    
    ax.set_xlabel('Importance Score')
    ax.set_title(f'{asset} - Top {top_n} Most Important Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    filepath = state.temp_dir / f"importance_{asset}_{model_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"importance_{asset}_{model_name}.png"
    )


@app.get("/charts/comparison/{asset}", tags=["Charts"])
async def chart_comparison(
    asset: str = Path(..., description="DOGE o TSLA")
):
    """Gráfico comparativo de modelos - Descarga PNG"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    # Evaluar todos los modelos
    evaluator = ModelEvaluator()
    models_metrics = {}
    
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'stacking']:
        pred = model.predict(state.test_df, model_name=model_name)
        true = state.test_df[target_col].values
        min_len = min(len(pred), len(true))
        
        result = evaluator.evaluate_regression(true[-min_len:], pred[-min_len:], model_name)
        models_metrics[model_name] = {
            'rmse': result['rmse'],
            'r2': result['r2'],
            'dir_acc': result['directional_accuracy']
        }
    
    # Crear gráfico
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(models_metrics.keys())
    
    # 1. RMSE
    rmse_values = [models_metrics[m]['rmse'] for m in models]
    axes[0].bar(models, rmse_values, color='#E63946')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title(f'{asset} - RMSE Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. R²
    r2_values = [models_metrics[m]['r2'] for m in models]
    axes[1].bar(models, r2_values, color='#2A9D8F')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title(f'{asset} - R² Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Directional Accuracy
    dir_acc_values = [models_metrics[m]['dir_acc'] * 100 for m in models]
    axes[2].bar(models, dir_acc_values, color='#F4A261')
    axes[2].set_ylabel('Directional Accuracy (%)')
    axes[2].set_title(f'{asset} - Dir. Accuracy Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(y=50, color='red', linestyle='--', linewidth=1, 
                   label='Random (50%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim([0, 100])
    
    plt.tight_layout()
    
    filepath = state.temp_dir / f"comparison_{asset}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"comparison_{asset}.png"
    )


# =================================================================
# ENDPOINTS - IMPACT CLASSIFIER
# =================================================================

@app.get("/impact/predict", tags=["Impact"])
async def predict_impact(
    n: int = Query(10, description="Número de predicciones", ge=1, le=100)
):
    """Clasificación de impacto de los últimos tweets"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    data = state.test_df.tail(n)
    predictions = state.impact_model.predict(data, model_name='xgboost')
    probas = state.impact_model.predict_proba(data, model_name='xgboost')
    
    class_names = ['No Impact', 'DOGE Only', 'TSLA Only', 'Both']
    
    results = []
    for i, (pred, proba) in enumerate(zip(predictions, probas)):
        results.append({
            'index': int(i),
            'predicted_class': class_names[pred],
            'class_id': int(pred),
            'probabilities': {
                class_names[j]: float(p) for j, p in enumerate(proba)
            }
        })
    
    return {
        "n_predictions": len(results),
        "predictions": results
    }


@app.post("/impact/predict-text", tags=["Impact"])
async def predict_impact_from_text(
    text: str = Query(..., description="Texto del tweet a analizar"),
    model_name: str = Query("xgboost", description="Modelo a usar")
):
    """
    🎯 Predice el impacto de un NUEVO TEXTO en DOGE y TSLA
    
    Toma un texto (ej: tweet de Elon Musk) y predice:
    - Probabilidad de impacto en DOGE
    - Probabilidad de impacto en TSLA
    - Magnitud esperada del impacto
    
    Ejemplo:
    ```
    POST /impact/predict-text?text=DOGE to the moon! 🚀
    ```
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Importar sentiment analyzer
    try:
        from src.sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
    except:
        # Fallback simple si no existe
        raise HTTPException(
            status_code=501,
            detail="SentimentAnalyzer no disponible. Necesitas el módulo de análisis de sentimiento."
        )
    
    # Analizar sentimiento del texto
    sentiment_result = analyzer.analyze_tweet(text)
    print(sentiment_result)
    
    # Crear dataframe con features mínimas necesarias
    # (Simulamos contexto de mercado actual)
    features_dict = {
        'sentiment_ensemble': sentiment_result.get('ensemble', 0.0),
        'relevance_score': sentiment_result.get('relevance', 0.5),
        'hour_sin': np.sin(2 * np.pi * datetime.now().hour / 24),
        'hour_cos': np.cos(2 * np.pi * datetime.now().hour / 24),
        'day_sin': np.sin(2 * np.pi * datetime.now().weekday() / 7),
        'day_cos': np.cos(2 * np.pi * datetime.now().weekday() / 7),
        # Features de mercado (usar valores medios del dataset)
        'doge_ret_1h': 0.0,
        'doge_vol_zscore': 0.0,
        'doge_rsi': 50.0,
        'tsla_ret_1h': 0.0,
        'tsla_market_open': 1 if 9 <= datetime.now().hour <= 16 else 0,
        'tsla_vol_zscore': 0.0,
        'mentions_doge': sentiment_result.get('mentions_doge', 0.0),
        'mentions_tesla': sentiment_result.get('mentions_tesla', 0.0),
        'relevance_score_lag1':1,
        'relevance_score_lag1':1
    }
    
    # Crear DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # Predecir
    prediction = state.impact_model.predict(input_df, model_name=model_name)[0]
    probabilities = state.impact_model.predict_proba(input_df, model_name=model_name)[0]
    
    class_names = ['No Impact', 'DOGE Only', 'TSLA Only', 'Both']
    
    # Calcular magnitud de impacto esperada
    # Usando los modelos DOGE y TSLA
    doge_impact = 0.0
    tsla_impact = 0.0
    
    if prediction in [1, 3]:  # DOGE impactado
        # Predecir retorno de DOGE
        try:
            doge_pred = state.doge_model.predict(input_df, model_name='stacking')[0]
            doge_impact = float(doge_pred)
        except:
            doge_impact = sentiment_result.get('ensemble', 0.0) * 0.02  # Estimación simple
    
    if prediction in [2, 3]:  # TSLA impactado
        try:
            tsla_pred = state.tsla_model.predict(input_df, model_name='stacking')[0]
            tsla_impact = float(tsla_pred)
        except:
            tsla_impact = sentiment_result.get('ensemble', 0.0) * 0.01
    
    return {
        "text": text,
        "sentiment": {
            "score": sentiment_result.get('ensemble', 0.0),
            "relevance": sentiment_result.get('relevance', 0.5)
        },
        "impact_prediction": {
            "predicted_class": class_names[prediction],
            "class_id": int(prediction),
            "probabilities": {
                "no_impact": f"{probabilities[0]*100:.2f}%",
                "doge_only": f"{probabilities[1]*100:.2f}%",
                "tsla_only": f"{probabilities[2]*100:.2f}%",
                "both": f"{probabilities[3]*100:.2f}%"
            }
        },
        "expected_impact": {
            "doge": {
                "affected": prediction in [1, 3],
                "expected_return_pct": f"{doge_impact*100:.2f}%",
                "magnitude": "high" if abs(doge_impact) > 0.02 else "medium" if abs(doge_impact) > 0.01 else "low"
            },
            "tsla": {
                "affected": prediction in [2, 3],
                "expected_return_pct": f"{tsla_impact*100:.2f}%",
                "magnitude": "high" if abs(tsla_impact) > 0.02 else "medium" if abs(tsla_impact) > 0.01 else "low"
            }
        },
        "recommendation": (
            "🚀 Alto impacto esperado" if prediction == 3 else
            "🐶 Impacto solo en DOGE" if prediction == 1 else
            "🚗 Impacto solo en TSLA" if prediction == 2 else
            "😴 Sin impacto significativo"
        )
    }

# =================================================================
# ENDPOINTS - PREPROCESSING CHARTS
# =================================================================
@app.get("/analysis/wordcloud", tags=["Analysis"])
async def generate_wordcloud():
    """
    📊 Genera WordCloud de las palabras más frecuentes de Elon Musk
    
    Devuelve imagen PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo (train + test)
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    analysis = exploratory_word_analysis(df, min_freq=15)
    
    # Crear WordCloud
    wc_config = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        max_words=200,
        relative_scaling=0.5,
        min_font_size=10
    )
    wordcloud = wc_config.generate_from_frequencies(analysis['word_freq'],)
    wc_image = wordcloud.to_image()

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_image, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Elon Musk Tweets - Most Frequent Words', fontsize=20, pad=20)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / "wordcloud_words.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename="wordcloud_elon_musk.png"
    )


@app.get("/analysis/wordcloud-bigrams", tags=["Analysis"])
async def generate_wordcloud_bigrams():
    """
    📊 Genera WordCloud de los bigramas más frecuentes
    
    Devuelve imagen PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    analysis = exploratory_word_analysis(df, min_freq=15)
    
    # Crear WordCloud
    wc_config = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='plasma',
            max_words=150,
            relative_scaling=0.5,
            min_font_size=10,
            regexp=None
    )
    wordcloud = wc_config.generate_from_frequencies(analysis['bigram_freq'])
    wc_image = wordcloud.to_image()
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_image, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Elon Musk Tweets - Most Frequent Bigrams', fontsize=20, pad=20)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / "wordcloud_bigrams.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename="wordcloud_bigrams_elon_musk.png"
    )


@app.get("/analysis/top-words", tags=["Analysis"])
async def top_words_chart(
    top_n: int = Query(30, description="Número de palabras a mostrar", ge=10, le=100)
):
    """
    📊 Top N palabras más usadas por Elon Musk
    
    Devuelve gráfico de barras PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    
    # Extraer palabras
    words = ' '.join(df['text_clean'].dropna().values)
    tokens = words.split()
    
    # Contar frecuencias
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(top_n)
    
    # Preparar datos
    words_list = [word for word, count in top_words]
    counts_list = [count for word, count in top_words]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words_list)))
    bars = ax.barh(words_list[::-1], counts_list[::-1], color=colors[::-1])
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Used Words by Elon Musk', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (bar, count) in enumerate(zip(bars, counts_list[::-1])):
        ax.text(count, i, f' {count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / f"top_{top_n}_words.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"top_{top_n}_words_elon_musk.png"
    )


@app.get("/analysis/top-bigrams", tags=["Analysis"])
async def top_bigrams_chart(
    top_n: int = Query(20, description="Número de bigramas a mostrar", ge=10, le=50)
):
    """
    📊 Top N bigramas más usados por Elon Musk
    
    Devuelve gráfico de barras PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    
    # Extraer bigramas
    words = ' '.join(df['text_clean'].dropna().values)
    tokens = words.split()
    bigrams_list = list(bigrams(tokens))
    
    # Contar frecuencias
    bigram_freq = Counter(bigrams_list)
    top_bigrams = bigram_freq.most_common(top_n)
    
    # Preparar datos
    bigrams_labels = [f"{w1} {w2}" for (w1, w2), count in top_bigrams]
    counts_list = [count for bigram, count in top_bigrams]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(bigrams_labels)))
    bars = ax.barh(bigrams_labels[::-1], counts_list[::-1], color=colors[::-1])
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Used Bigrams by Elon Musk', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (bar, count) in enumerate(zip(bars, counts_list[::-1])):
        ax.text(count, i, f' {count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / f"top_{top_n}_bigrams.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"top_{top_n}_bigrams_elon_musk.png"
    )


@app.get("/analysis/stats", tags=["Analysis"])
async def text_statistics():
    """
    📊 Estadísticas generales del texto de Elon Musk
    
    Retorna JSON con métricas
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    print(df.head(10))

    analysis = exploratory_word_analysis(df, min_freq=15)
    df_musk = analysis['df_clean']
    words = ' '.join(df_musk['text_clean'].dropna().values)

    total_words = len(words)
    return {
        "total_tweets": len(analysis['df_clean']),
        "total_words": total_words,
        "top_10_words": analysis['top_words'],
        "top_10_bigrams": analysis['top_bigrams']
    }
# =================================================================
# ERROR HANDLERS
# =================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    import traceback
    traceback.print_exc()  # Log completo en consola
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()  # Serializable
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)