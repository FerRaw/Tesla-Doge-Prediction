"""
Clase base abstracta para todos los modelos predictivos
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import joblib
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class BasePredictor(ABC):
    """Clase base para modelos de predicción"""
    
    def __init__(self, model_name: str, version: str = "v1"):
        self.model_name = model_name
        self.version = version
        self.models = {}
        self.metrics = {}
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.ensemble_weights = {}
        
        self.metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'trained_at': None,
            'training_samples': None,
            'features_count': None,
            'cv_splits': None
        }
    
    @abstractmethod
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepara features y target para el modelo
        
        Args:
            df: DataFrame con datos
            is_train: Si es entrenamiento (necesita target)
        
        Returns:
            (X_scaled, y) si is_train=True
            (X_scaled, None) si is_train=False
        """
        pass
    
    @abstractmethod
    def train(self, df: pd.DataFrame, **kwargs):
        """Entrena todos los modelos"""
        pass
    
    @abstractmethod
    def predict(
        self, 
        df: pd.DataFrame, 
        model_name: str = 'ensemble'
    ) -> np.ndarray:
        """
        Realiza predicciones
        
        Args:
            df: DataFrame con features
            model_name: Modelo a usar ('ensemble', 'xgboost', etc.)
        
        Returns:
            Array de predicciones
        """
        pass
    
    def save(self, path: Path):
        """Guarda modelo y metadata"""
        # Asegurar que el directorio existe
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo
        joblib.dump(self, path)
        
        # Guardar metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"✅ Modelo guardado: {path}")
        print(f"✅ Metadata guardada: {metadata_path}")
    
    @classmethod
    def load(cls, path: Path):
        """Carga modelo desde archivo"""
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        
        print(f"📂 Cargando modelo desde {path}...")
        model = joblib.load(path)
        print(f"✅ Modelo cargado: {model.model_name}")
        return model
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> Dict[str, float]:
        """Obtiene importancia de features"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))
        else:
            print(f"⚠️ Modelo {model_name} no tiene feature_importances_")
            return {}
    
    def update_metadata(self, **kwargs):
        """Actualiza metadata del modelo"""
        self.metadata.update(kwargs)
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Retorna resumen del modelo"""
        return {
            'name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'models_available': list(self.models.keys()),
            'metrics': self.metrics,
            'ensemble_weights': self.ensemble_weights,
            'metadata': self.metadata
        }
    
    def _create_ensemble_weights(self):
        """
        Crea pesos para ensemble basados en RMSE de CV
        Pesos inversamente proporcionales al error
        """
        weights = {}
        total_inv_rmse = 0
        
        for name in ['xgboost', 'lightgbm', 'elastic_net', 'lstm']:
            if name in self.metrics:
                inv_rmse = 1 / self.metrics[name]['cv_rmse_mean']
                weights[name] = inv_rmse
                total_inv_rmse += inv_rmse
        
        # Normalizar
        self.ensemble_weights = {
            k: v / total_inv_rmse 
            for k, v in weights.items()
        }
        
        print(f"\n⚖️  Ensemble Weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"   {name}: {weight:.4f}")
    
    def _print_cv_results(self, model_name: str, scores: list):
        """Imprime resultados de CV de forma organizada"""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\n   {model_name.upper()}:")
        print(f"      CV RMSE: {mean_score:.6f} (±{std_score:.6f})")
        print(f"      Min: {min(scores):.6f} | Max: {max(scores):.6f}")