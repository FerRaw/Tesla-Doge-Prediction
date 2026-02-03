"""
FastAPI Application - API de Predicción TFM
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config.settings import settings
from src.api.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    DogePredictionResponse,
    TslaPredictionResponse,
    ImpactPredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelsInfoResponse
)
from src.models.predictors import DOGEPredictor, TSLAPredictor, ImpactClassifier


# =============================================================================
# Model Registry
# =============================================================================

class ModelRegistry:
    """Registro global de modelos cargados"""
    
    def __init__(self):
        self.doge_predictor = None
        self.tsla_predictor = None
        self.impact_classifier = None
        self.models_loaded = {
            "doge": False,
            "tsla": False,
            "impact": False
        }
    
    def load_models(self):
        """Carga modelos en memoria"""
        print("🚀 Cargando modelos en memoria...")
        
        # DOGE
        doge_path = settings.MODELS_DIR / f"doge_predictor_{settings.MODELS_VERSION}.pkl"
        if doge_path.exists():
            try:
                self.doge_predictor = DOGEPredictor.load(doge_path)
                self.models_loaded["doge"] = True
                print("   ✅ Modelo DOGE cargado")
            except Exception as e:
                print(f"   ⚠️ Error cargando DOGE: {e}")
        else:
            print(f"   ⚠️ Modelo DOGE no encontrado: {doge_path}")
        
        # TSLA
        tsla_path = settings.MODELS_DIR / f"tsla_predictor_{settings.MODELS_VERSION}.pkl"
        if tsla_path.exists():
            try:
                self.tsla_predictor = TSLAPredictor.load(tsla_path)
                self.models_loaded["tsla"] = True
                print("   ✅ Modelo TSLA cargado")
            except Exception as e:
                print(f"   ⚠️ Error cargando TSLA: {e}")
        else:
            print(f"   ⚠️ Modelo TSLA no encontrado: {tsla_path}")
        
        # Impact
        impact_path = settings.MODELS_DIR / f"impact_classifier_{settings.MODELS_VERSION}.pkl"
        if impact_path.exists():
            try:
                self.impact_classifier = ImpactClassifier.load(impact_path)
                self.models_loaded["impact"] = True
                print("   ✅ Clasificador de Impacto cargado")
            except Exception as e:
                print(f"   ⚠️ Error cargando Impact: {e}")
        else:
            print(f"   ⚠️ Clasificador no encontrado: {impact_path}")
        
        loaded_count = sum(self.models_loaded.values())
        print(f"✅ {loaded_count}/3 modelos cargados")


# Instancia global
model_registry = ModelRegistry()


# =============================================================================
# Lifespan Events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eventos de inicio y cierre"""
    # Startup
    print("="*70)
    print("🚀 INICIANDO API DE PREDICCIÓN - TFM")
    print("="*70)
    
    model_registry.load_models()
    
    print("✅ API lista para recibir requests")
    print(f"📍 Documentación: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("="*70)
    
    yield
    
    # Shutdown
    print("🛑 Cerrando API...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title=settings.API_TITLE,
    description="Sistema de predicción de impacto de tweets de Elon Musk en DOGE y TSLA",
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def features_to_dataframe(features) -> pd.DataFrame:
    """Convierte Pydantic model a DataFrame"""
    data = features.model_dump()
    return pd.DataFrame([data])


def get_impact_label(impact_class: int) -> str:
    """Convierte clase numérica a etiqueta"""
    labels = {
        0: "Sin impacto significativo",
        1: "Impacto en DOGE",
        2: "Impacto en TSLA",
        3: "Impacto en ambos (DOGE y TSLA)"
    }
    return labels.get(impact_class, "Desconocido")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz"""
    return {
        "service": "TFM - Musk Tweets Market Impact Prediction API",
        "version": settings.API_VERSION,
        "status": "running",
        "models_loaded": model_registry.models_loaded,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predictions": {
                "doge": "/predict/doge",
                "tesla": "/predict/tesla",
                "impact": "/predict/impact",
                "batch": "/predict/batch/{asset}"
            },
            "models_info": "/models/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica estado de salud"""
    return HealthResponse(
        status="healthy" if any(model_registry.models_loaded.values()) else "degraded",
        models_loaded=model_registry.models_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/doge", response_model=DogePredictionResponse, tags=["Predictions"])
async def predict_doge(request: PredictionRequest):
    """
    Predice retorno futuro de DOGECOIN
    
    - **features**: Características del mercado y sentimiento
    - **model_type**: Modelo a usar (ensemble, xgboost, lightgbm, lstm, elastic_net)
    
    Retorna el retorno predicho en formato decimal (0.02 = 2%)
    """
    if not model_registry.models_loaded["doge"]:
        raise HTTPException(
            status_code=503,
            detail="Modelo DOGE no disponible. Entrena el modelo primero."
        )
    
    try:
        # Convertir features a DataFrame
        df = features_to_dataframe(request.features)
        
        # Predicción
        prediction = model_registry.doge_predictor.predict(
            df,
            model_name=request.model_type
        )[0]
        
        # Intervalo de confianza (simplificado - basado en std de CV)
        std_dev = 0.01  # Placeholder
        confidence_interval = {
            "lower": float(prediction - 1.96 * std_dev),
            "upper": float(prediction + 1.96 * std_dev)
        }
        
        return DogePredictionResponse(
            predicted_return=float(prediction),
            prediction_timestamp=datetime.now().isoformat(),
            model_used=request.model_type,
            confidence_interval=confidence_interval
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción DOGE: {str(e)}"
        )


@app.post("/predict/tesla", response_model=TslaPredictionResponse, tags=["Predictions"])
async def predict_tesla(request: PredictionRequest):
    """
    Predice retorno futuro de acciones TESLA
    
    Similar a /predict/doge pero para TSLA
    """
    if not model_registry.models_loaded["tsla"]:
        raise HTTPException(
            status_code=503,
            detail="Modelo TSLA no disponible"
        )
    
    try:
        df = features_to_dataframe(request.features)
        
        prediction = model_registry.tsla_predictor.predict(
            df,
            model_name=request.model_type
        )[0]
        
        std_dev = 0.008
        confidence_interval = {
            "lower": float(prediction - 1.96 * std_dev),
            "upper": float(prediction + 1.96 * std_dev)
        }
        
        return TslaPredictionResponse(
            predicted_return=float(prediction),
            prediction_timestamp=datetime.now().isoformat(),
            model_used=request.model_type,
            confidence_interval=confidence_interval
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción TSLA: {str(e)}"
        )


@app.post("/predict/impact", response_model=ImpactPredictionResponse, tags=["Predictions"])
async def predict_impact(request: PredictionRequest):
    """
    Clasifica el impacto potencial de un tweet
    
    Retorna:
    - **impact_class**: 0 (Sin impacto), 1 (DOGE), 2 (TSLA), 3 (Ambos)
    - **probabilities**: Probabilidad de cada clase
    """
    if not model_registry.models_loaded["impact"]:
        raise HTTPException(
            status_code=503,
            detail="Clasificador de impacto no disponible"
        )
    
    try:
        df = features_to_dataframe(request.features)
        
        # Predicción
        impact_class = model_registry.impact_classifier.predict(df)[0]
        probas = model_registry.impact_classifier.predict_proba(df)[0]
        
        probabilities = {
            "no_impact": float(probas[0]),
            "doge_impact": float(probas[1]) if len(probas) > 1 else 0.0,
            "tsla_impact": float(probas[2]) if len(probas) > 2 else 0.0,
            "both_impact": float(probas[3]) if len(probas) > 3 else 0.0
        }
        
        return ImpactPredictionResponse(
            impact_class=int(impact_class),
            impact_label=get_impact_label(int(impact_class)),
            probabilities=probabilities,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en clasificación: {str(e)}"
        )


@app.post("/predict/batch/{asset}", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(asset: str, request: BatchPredictionRequest):
    """
    Predicción batch para múltiples observaciones
    
    - **asset**: 'doge' o 'tesla'
    - **features_list**: Lista de features
    
    Útil para backtesting
    """
    asset = asset.lower()
    
    if asset not in ["doge", "tesla"]:
        raise HTTPException(
            status_code=400,
            detail="Asset debe ser 'doge' o 'tesla'"
        )
    
    model_key = "doge" if asset == "doge" else "tsla"
    
    if not model_registry.models_loaded[model_key]:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo {asset.upper()} no disponible"
        )
    
    try:
        # Convertir lista a DataFrame
        data = [f.model_dump() for f in request.features_list]
        df = pd.DataFrame(data)
        
        # Predicción
        if asset == "doge":
            predictions = model_registry.doge_predictor.predict(df, request.model_type)
        else:
            predictions = model_registry.tsla_predictor.predict(df, request.model_type)
        
        return BatchPredictionResponse(
            asset=asset.upper(),
            predictions=predictions.tolist(),
            count=len(predictions),
            model_used=request.model_type,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción batch: {str(e)}"
        )


@app.get("/models/info", response_model=ModelsInfoResponse, tags=["Models"])
async def get_models_info():
    """
    Información sobre los modelos cargados
    
    Retorna métricas, pesos de ensemble y feature names
    """
    info = {}
    
    if model_registry.models_loaded["doge"]:
        model = model_registry.doge_predictor
        info["doge"] = ModelInfoResponse(
            model_name=model.model_name,
            version=model.version,
            is_trained=model.is_trained,
            models_available=list(model.models.keys()),
            metrics=model.metrics,
            ensemble_weights=model.ensemble_weights,
            feature_names=model.feature_names
        )
    
    if model_registry.models_loaded["tsla"]:
        model = model_registry.tsla_predictor
        info["tsla"] = ModelInfoResponse(
            model_name=model.model_name,
            version=model.version,
            is_trained=model.is_trained,
            models_available=list(model.models.keys()),
            metrics=model.metrics,
            ensemble_weights=model.ensemble_weights,
            feature_names=model.feature_names
        )
    
    if model_registry.models_loaded["impact"]:
        model = model_registry.impact_classifier
        info["impact"] = ModelInfoResponse(
            model_name=model.model_name,
            version=model.version,
            is_trained=model.is_trained,
            models_available=list(model.models.keys()),
            metrics=model.metrics,
            feature_names=model.feature_names
        )
    
    return ModelsInfoResponse(**info)