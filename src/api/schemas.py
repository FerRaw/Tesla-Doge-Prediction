"""
Pydantic schemas para requests y responses de la API
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


# =============================================================================
# REQUEST MODELS
# =============================================================================

class MarketFeatures(BaseModel):
    """Features de mercado para predicción"""
    
    # DOGE features
    doge_ret_1h: Optional[float] = Field(None, description="Retorno 1h DOGE")
    doge_vol_zscore: Optional[float] = Field(None, description="Z-score volatilidad DOGE")
    doge_buy_pressure: Optional[float] = Field(None, description="Presión compradora DOGE")
    doge_rsi: Optional[float] = Field(None, description="RSI DOGE")
    
    # TSLA features
    tsla_ret_1h: Optional[float] = Field(None, description="Retorno 1h TSLA")
    tsla_market_open: Optional[int] = Field(None, description="Mercado abierto (1/0)")
    tsla_vol_zscore: Optional[float] = Field(None, description="Z-score volatilidad TSLA")
    
    # Sentiment features
    sentiment_ensemble: Optional[float] = Field(None, description="Sentimiento ensemble [-1, 1]")
    relevance_score: Optional[float] = Field(None, description="Score de relevancia [0, 100]")
    
    # Sentiment lags
    sentiment_ensemble_lag1: Optional[float] = Field(None, description="Sentimiento lag 1h")
    sentiment_ensemble_lag2: Optional[float] = Field(None, description="Sentimiento lag 2h")
    sentiment_ensemble_lag3: Optional[float] = Field(None, description="Sentimiento lag 3h")
    relevance_score_lag1: Optional[float] = Field(None, description="Relevancia lag 1h")
    relevance_score_lag2: Optional[float] = Field(None, description="Relevancia lag 2h")
    relevance_score_lag3: Optional[float] = Field(None, description="Relevancia lag 3h")
    
    # Temporal features
    hour_sin: float = Field(..., description="Hora (sin)")
    hour_cos: float = Field(..., description="Hora (cos)")
    day_sin: float = Field(..., description="Día semana (sin)")
    day_cos: float = Field(..., description="Día semana (cos)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doge_ret_1h": 0.015,
                "doge_vol_zscore": 1.2,
                "doge_buy_pressure": 0.62,
                "doge_rsi": 58.3,
                "tsla_ret_1h": 0.008,
                "tsla_market_open": 1,
                "tsla_vol_zscore": 0.5,
                "sentiment_ensemble": 0.75,
                "relevance_score": 85.0,
                "sentiment_ensemble_lag1": 0.70,
                "sentiment_ensemble_lag2": 0.65,
                "sentiment_ensemble_lag3": 0.60,
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "day_sin": 0.0,
                "day_cos": 1.0
            }
        }


class PredictionRequest(BaseModel):
    """Request para predicción única"""
    features: MarketFeatures
    model_type: str = Field(
        "ensemble",
        description="Tipo de modelo: ensemble, xgboost, lightgbm, lstm, elastic_net"
    )


class BatchPredictionRequest(BaseModel):
    """Request para predicción batch"""
    features_list: List[MarketFeatures]
    model_type: str = Field("ensemble", description="Tipo de modelo")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class PredictionResponse(BaseModel):
    """Respuesta de predicción genérica"""
    predicted_return: float = Field(..., description="Retorno predicho")
    prediction_timestamp: str
    model_used: str
    confidence_interval: Optional[Dict[str, float]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_return": 0.0234,
                "prediction_timestamp": "2026-01-17T10:30:00",
                "model_used": "ensemble",
                "confidence_interval": {
                    "lower": 0.015,
                    "upper": 0.032
                }
            }
        }


class DogePredictionResponse(PredictionResponse):
    """Respuesta específica para DOGE"""
    asset: str = "DOGE"


class TslaPredictionResponse(PredictionResponse):
    """Respuesta específica para TSLA"""
    asset: str = "TSLA"


class ImpactPredictionResponse(BaseModel):
    """Respuesta de clasificación de impacto"""
    impact_class: int = Field(..., description="0: Sin impacto, 1: DOGE, 2: TSLA, 3: Ambos")
    impact_label: str
    probabilities: Dict[str, float]
    prediction_timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "impact_class": 1,
                "impact_label": "Impacto en DOGE",
                "probabilities": {
                    "no_impact": 0.15,
                    "doge_impact": 0.68,
                    "tsla_impact": 0.10,
                    "both_impact": 0.07
                },
                "prediction_timestamp": "2026-01-17T10:30:00"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción batch"""
    asset: str
    predictions: List[float]
    count: int
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Información de un modelo"""
    model_name: str
    version: str
    is_trained: bool
    models_available: List[str]
    metrics: Dict
    ensemble_weights: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None


class ModelsInfoResponse(BaseModel):
    """Información de todos los modelos"""
    doge: Optional[ModelInfoResponse] = None
    tsla: Optional[ModelInfoResponse] = None
    impact: Optional[ModelInfoResponse] = None