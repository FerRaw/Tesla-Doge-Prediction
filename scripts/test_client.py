"""
Cliente de ejemplo para testear la API

Uso:
    python scripts/test_api_client.py
"""

import requests
import json
import math
from datetime import datetime


class PredictionClient:
    """Cliente para interactuar con la API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Health check"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def create_temporal_features(self, timestamp: datetime = None):
        """Crea features temporales cíclicas"""
        if timestamp is None:
            timestamp = datetime.now()
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        return {
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "day_sin": math.sin(2 * math.pi * day_of_week / 7),
            "day_cos": math.cos(2 * math.pi * day_of_week / 7)
        }
    
    def predict_doge(self, features: dict, model_type: str = "ensemble"):
        """Predice retorno de DOGE"""
        payload = {
            "features": features,
            "model_type": model_type
        }
        
        response = requests.post(
            f"{self.base_url}/predict/doge",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def predict_tesla(self, features: dict, model_type: str = "ensemble"):
        """Predice retorno de TSLA"""
        payload = {
            "features": features,
            "model_type": model_type
        }
        
        response = requests.post(
            f"{self.base_url}/predict/tesla",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def predict_impact(self, features: dict):
        """Clasifica impacto de tweet"""
        payload = {
            "features": features,
            "model_type": "xgboost"
        }
        
        response = requests.post(
            f"{self.base_url}/predict/impact",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def get_models_info(self):
        """Obtiene información de modelos"""
        response = requests.get(f"{self.base_url}/models/info")
        return response.json()


# =============================================================================
# EJEMPLOS DE USO
# =============================================================================

def example_1_doge_prediction():
    """Ejemplo 1: Predicción de DOGE"""
    print("\n" + "="*70)
    print("EJEMPLO 1: Predicción de retorno DOGE")
    print("="*70)
    
    client = PredictionClient()
    
    # Crear features de ejemplo
    temporal = client.create_temporal_features()
    
    features = {
        # DOGE
        "doge_ret_1h": 0.015,
        "doge_vol_zscore": 1.2,
        "doge_buy_pressure": 0.62,
        "doge_rsi": 58.3,
        
        # TSLA (opcional para DOGE, pero incluimos por completitud)
        "tsla_ret_1h": 0.008,
        "tsla_market_open": 1,
        "tsla_vol_zscore": 0.5,
        
        # Sentiment
        "sentiment_ensemble": 0.75,
        "relevance_score": 85.0,
        
        # Sentiment lags
        "sentiment_ensemble_lag1": 0.70,
        "sentiment_ensemble_lag2": 0.65,
        "sentiment_ensemble_lag3": 0.60,
        "relevance_score_lag1": 80.0,
        "relevance_score_lag2": 75.0,
        "relevance_score_lag3": 70.0,
        
        # Temporal
        **temporal
    }
    
    # Predecir
    result = client.predict_doge(features, model_type="ensemble")
    
    print(f"\n📊 RESULTADO:")
    print(f"   Retorno predicho: {result['predicted_return']:.4f} ({result['predicted_return']*100:.2f}%)")
    print(f"   Modelo usado: {result['model_used']}")
    print(f"   Timestamp: {result['prediction_timestamp']}")
    
    if result.get('confidence_interval'):
        ci = result['confidence_interval']
        print(f"   Intervalo confianza 95%: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    
    # Interpretar
    if result['predicted_return'] > 0.01:
        print(f"\n✅ SEÑAL ALCISTA: Se espera subida > 1%")
    elif result['predicted_return'] < -0.01:
        print(f"\n⚠️ SEÑAL BAJISTA: Se espera caída > 1%")
    else:
        print(f"\n➡️ SEÑAL NEUTRAL: Movimiento esperado < 1%")


def example_2_tesla_prediction():
    """Ejemplo 2: Predicción de TSLA"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Predicción de retorno TESLA")
    print("="*70)
    
    client = PredictionClient()
    temporal = client.create_temporal_features()
    
    features = {
        # TSLA
        "tsla_ret_1h": 0.008,
        "tsla_market_open": 1,
        "tsla_vol_zscore": 0.5,
        
        # Sentiment (con lag 1h optimizado para TSLA)
        "sentiment_ensemble": 0.85,
        "relevance_score": 90.0,
        "sentiment_ensemble_lag1": 0.80,
        "relevance_score_lag1": 85.0,
        
        # Temporal
        **temporal
    }
    
    result = client.predict_tesla(features, model_type="ensemble")
    
    print(f"\n📊 RESULTADO:")
    print(f"   Retorno predicho: {result['predicted_return']:.4f} ({result['predicted_return']*100:.2f}%)")
    print(f"   Modelo usado: {result['model_used']}")
    
    # Calcular precio estimado
    current_price = 350.0  # Precio hipotético
    predicted_price = current_price * (1 + result['predicted_return'])
    print(f"\n💰 Si precio actual = ${current_price:.2f}")
    print(f"   Precio estimado próxima hora: ${predicted_price:.2f}")


def example_3_impact_classification():
    """Ejemplo 3: Clasificación de impacto"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Clasificar impacto de tweet")
    print("="*70)
    
    client = PredictionClient()
    temporal = client.create_temporal_features()
    
    # Escenario: Tweet muy positivo sobre Dogecoin
    features = {
        "doge_vol_zscore": 0.8,
        "tsla_vol_zscore": 0.3,
        "tsla_market_open": 1,
        
        "sentiment_ensemble": 0.9,
        "relevance_score": 95.0,
        
        **temporal
    }
    
    result = client.predict_impact(features)
    
    print(f"\n📊 RESULTADO:")
    print(f"   Clase predicha: {result['impact_class']}")
    print(f"   Etiqueta: {result['impact_label']}")
    print(f"\n   Probabilidades:")
    for label, prob in result['probabilities'].items():
        print(f"      {label:15s}: {prob:.2%}")
    
    # Recomendación
    print(f"\n💡 RECOMENDACIÓN:")
    if result['impact_class'] == 1:
        print(f"   📈 Monitorear DOGECOIN - Impacto esperado")
    elif result['impact_class'] == 2:
        print(f"   🚗 Monitorear TESLA - Impacto esperado")
    elif result['impact_class'] == 3:
        print(f"   ⚠️ ¡ALERTA! Impacto esperado en AMBOS activos")
    else:
        print(f"   ➡️ Bajo impacto esperado")


def example_4_models_info():
    """Ejemplo 4: Información de modelos"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Información de modelos")
    print("="*70)
    
    client = PredictionClient()
    info = client.get_models_info()
    
    print("\n📊 MODELOS CARGADOS:")
    
    if 'doge' in info and info['doge']:
        print(f"\n🐕 DOGE PREDICTOR:")
        print(f"   Versión: {info['doge']['version']}")
        print(f"   Modelos disponibles: {', '.join(info['doge']['models_available'])}")
        
        if 'ensemble_weights' in info['doge'] and info['doge']['ensemble_weights']:
            print(f"\n   Pesos Ensemble:")
            for model, weight in info['doge']['ensemble_weights'].items():
                print(f"      {model:15s}: {weight:.4f}")
    
    if 'tsla' in info and info['tsla']:
        print(f"\n🚗 TSLA PREDICTOR:")
        print(f"   Versión: {info['tsla']['version']}")
        print(f"   Modelos disponibles: {', '.join(info['tsla']['models_available'])}")
    
    if 'impact' in info and info['impact']:
        print(f"\n🎯 IMPACT CLASSIFIER:")
        print(f"   Versión: {info['impact']['version']}")


def main():
    """Ejecuta ejemplos"""
    print("="*70)
    print("🎓 CLIENTE DE TESTING - API DE PREDICCIÓN TFM")
    print("="*70)
    print("\nAsegúrate de que la API esté corriendo en http://localhost:8000")
    print("(Ejecuta: python scripts/03_run_api.py)")
    
    try:
        # Health check
        client = PredictionClient()
        health = client.health_check()
        print(f"\n✅ API disponible - Estado: {health['status']}")
        print(f"   Modelos cargados: {health['models_loaded']}")
        
        # Ejecutar ejemplos
        example_1_doge_prediction()
        example_2_tesla_prediction()
        example_3_impact_classification()
        example_4_models_info()
        
        print("\n" + "="*70)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se puede conectar a la API")
        print("   Ejecuta primero: python scripts/03_run_api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()