"""
Predictores Mejorados FINALES - Solo Boosting (Máximo Rendimiento)

Incluye únicamente los modelos que demostraron funcionar:
- XGBoost (mejor directional accuracy)
- LightGBM (buena generalización)  
- CatBoost (mejor RMSE y R²)
- Stacking optimizado

Deep Learning descartado por overfitting severo.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.ensemble import StackingRegressor

# Boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Config
from config.settings import settings
from src.models.base_predictor import BasePredictor


# =================================================================
# IMPROVED DOGE PREDICTOR
# =================================================================
class ImprovedDOGEPredictor(BasePredictor):
    """
    Predictor DOGE Mejorado - Solo modelos que funcionan
    
    Performance en Test (esperado):
    - CatBoost:  RMSE 0.011, R² 0.22, Dir 65%
    - XGBoost:   RMSE 0.011, R² 0.21, Dir 66%
    - LightGBM:  RMSE 0.011, R² 0.20, Dir 66%
    - Stacking:  RMSE 0.011, R² 0.22, Dir 65%
    """
    
    def __init__(self, version: str = "v2", use_advanced_models: bool = False):
        super().__init__("doge_predictor_improved", version)
        self.use_advanced_models = use_advanced_models  # Siempre False en versión final
        self.scaler = StandardScaler()
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features incluyendo avanzadas"""
        
        # Features básicas + avanzadas
        feature_cols = [
            # Market features
            'doge_ret_1h', 'doge_vol_zscore', 'doge_buy_pressure', 'doge_rsi',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            
            # Sentiment
            'sentiment_ensemble', 'relevance_score',
            'sentiment_ensemble_lag1', 'sentiment_ensemble_lag2', 'sentiment_ensemble_lag3',
            'relevance_score_lag1', 'relevance_score_lag2', 'relevance_score_lag3',
        ]
        
        # Añadir features avanzadas si existen
        advanced_keywords = [
            'wavelet', 'autocorr', 'corr_', 'beta_', 'vol_ratio',
            'momentum_divergence', 'sentiment_x_vol', 'sentiment_velocity',
            'sentiment_acceleration', 'relevance_conditional',
            'vol_regime', 'atr', 'session_', 'is_weekend', 'momentum_'
        ]
        
        for col in df.columns:
            if any(keyword in col for keyword in advanced_keywords):
                if col not in feature_cols:
                    feature_cols.append(col)
        
        # Filtrar columnas disponibles
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if is_train:
            self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        y = df['TARGET_DOGE'].values if is_train and 'TARGET_DOGE' in df.columns else None
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena solo modelos de boosting optimizados"""
        print("\n" + "="*70)
        print(f"🚀 ENTRENANDO {self.model_name.upper()} (VERSIÓN MEJORADA)")
        print("="*70)
        
        X, y = self.prepare_features(df, is_train=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # =================================================================
        # 1. XGBoost Mejorado
        # =================================================================
        print("\n📊 [1/4] Entrenando XGBoost Mejorado...")
        xgb_scores = []
        
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=settings.RANDOM_SEED,
            tree_method='hist',
            verbosity=0
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['xgboost'].fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False
            )
            pred = self.models['xgboost'].predict(X[val_idx])
            xgb_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['xgboost'] = {
            'cv_rmse_mean': np.mean(xgb_scores),
            'cv_rmse_std': np.std(xgb_scores),
            'cv_rmse_min': np.min(xgb_scores),
            'cv_rmse_max': np.max(xgb_scores)
        }
        self.models['xgboost'].fit(X, y)
        self._print_cv_results('xgboost', xgb_scores)
        
        # =================================================================
        # 2. LightGBM Mejorado
        # =================================================================
        print("\n📊 [2/4] Entrenando LightGBM Mejorado...")
        lgb_scores = []
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=settings.RANDOM_SEED,
            verbose=-1
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['lightgbm'].fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            pred = self.models['lightgbm'].predict(X[val_idx])
            lgb_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['lightgbm'] = {
            'cv_rmse_mean': np.mean(lgb_scores),
            'cv_rmse_std': np.std(lgb_scores),
            'cv_rmse_min': np.min(lgb_scores),
            'cv_rmse_max': np.max(lgb_scores)
        }
        self.models['lightgbm'].fit(X, y)
        self._print_cv_results('lightgbm', lgb_scores)
        
        # =================================================================
        # 3. CatBoost
        # =================================================================
        print("\n📊 [3/4] Entrenando CatBoost...")
        cat_scores = []
        
        self.models['catboost'] = cb.CatBoostRegressor(
            iterations=500,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=settings.RANDOM_SEED,
            verbose=False
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['catboost'].fit(
                X[train_idx], y[train_idx],
                eval_set=(X[val_idx], y[val_idx]),
                verbose=False
            )
            pred = self.models['catboost'].predict(X[val_idx])
            cat_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['catboost'] = {
            'cv_rmse_mean': np.mean(cat_scores),
            'cv_rmse_std': np.std(cat_scores),
            'cv_rmse_min': np.min(cat_scores),
            'cv_rmse_max': np.max(cat_scores)
        }
        self.models['catboost'].fit(X, y)
        self._print_cv_results('catboost', cat_scores)
        
        # =================================================================
        # 4. Stacking Ensemble Optimizado
        # =================================================================
        print("\n📊 [4/4] Creando Stacking Ensemble Optimizado...")
        self._create_stacking_ensemble(X, y, tscv)
        
        self.is_trained = True
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO MEJORADO COMPLETADO")
        print("="*70)
    
    def _create_stacking_ensemble(self, X, y, tscv):
        """Crea ensemble optimizado"""
        base_models = [
            ('catboost', self.models['catboost']),
            ('xgboost', self.models['xgboost']),
            ('lightgbm', self.models['lightgbm'])
        ]
        
        meta_learner = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            random_state=settings.RANDOM_SEED
        )
        
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            passthrough=True,
            n_jobs=-1
        )
        
        # Evaluar con CV
        stacking_scores = []
        for train_idx, val_idx in tscv.split(X):
            stacking.fit(X[train_idx], y[train_idx])
            pred = stacking.predict(X[val_idx])
            stacking_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        stacking.fit(X, y)
        self.models['stacking'] = stacking
        
        self.metrics['stacking'] = {
            'cv_rmse_mean': np.mean(stacking_scores),
            'cv_rmse_std': np.std(stacking_scores),
            'cv_rmse_min': np.min(stacking_scores),
            'cv_rmse_max': np.max(stacking_scores)
        }
        
        self._print_cv_results('stacking', stacking_scores)
    
    def predict(self, df: pd.DataFrame, model_name: str = 'stacking') -> np.ndarray:
        """Predice con modelo especificado"""
        X, _ = self.prepare_features(df, is_train=False)
        return self.models[model_name].predict(X)


# =================================================================
# IMPROVED TSLA PREDICTOR
# =================================================================
class ImprovedTSLAPredictor(ImprovedDOGEPredictor):
    """Predictor TSLA Mejorado - Hereda de DOGE con features TSLA"""
    
    def __init__(self, version: str = "v2", use_advanced_models: bool = False):
        BasePredictor.__init__(self, "tsla_predictor_improved", version)
        self.use_advanced_models = use_advanced_models
        self.scaler = StandardScaler()
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features para TSLA"""
        
        feature_cols = [
            'tsla_ret_1h', 'tsla_market_open', 'tsla_vol_zscore',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Sentimiento con lag óptimo (1h según Granger)
        sentiment_features = [
            'sentiment_ensemble', 'relevance_score',
            'sentiment_ensemble_lag1', 'relevance_score_lag1'
        ]
        
        for feat in sentiment_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Features opcionales
        optional_features = [
            'tsla_momentum_3h', 'tsla_momentum_6h', 'tsla_momentum_12h'
        ]
        
        for feat in optional_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Features avanzadas
        advanced_keywords = [
            'wavelet', 'autocorr', 'corr_', 'beta_', 'vol_ratio',
            'momentum_divergence', 'sentiment_x_vol', 'sentiment_velocity',
            'vol_regime', 'atr'
        ]
        
        for col in df.columns:
            if any(keyword in col for keyword in advanced_keywords):
                if col not in feature_cols:
                    feature_cols.append(col)
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if is_train:
            self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y = df['TARGET_TSLA'].values if 'TARGET_TSLA' in df.columns else None
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena TSLA"""
        print("\n" + "="*70)
        print("🚀 ENTRENAMIENTO MEJORADO - TSLA PREDICTOR")
        print("="*70)
        
        ImprovedDOGEPredictor.train(self, df, n_splits)
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO TSLA COMPLETADO")
        print("="*70)


# =================================================================
# IMPACT CLASSIFIER
# =================================================================
class ImpactClassifier(BasePredictor):
    """Clasificador de impacto de tweets"""
    
    def __init__(self, version: str = "v1"):
        super().__init__("impact_classifier", version)
        self.impact_threshold = 0.02
        self.scaler = StandardScaler()
    
    def create_impact_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels de impacto (4 clases)"""
        df = df.copy()
        
        doge_impact = abs(df['TARGET_DOGE']) > self.impact_threshold
        tsla_impact = abs(df['TARGET_TSLA']) > self.impact_threshold
        
        df['impact_class'] = 0  # Sin impacto
        df.loc[doge_impact & ~tsla_impact, 'impact_class'] = 1  # Solo DOGE
        df.loc[~doge_impact & tsla_impact, 'impact_class'] = 2  # Solo TSLA
        df.loc[doge_impact & tsla_impact, 'impact_class'] = 3   # Ambos
        
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features para clasificación"""
        
        feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        # Sentimiento
        sentiment_features = [
            'sentiment_ensemble', 'relevance_score',
            'sentiment_ensemble_lag1', 'relevance_score_lag1'
        ]
        for feat in sentiment_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Contexto de mercado
        market_features = [
            'doge_ret_1h', 'doge_vol_zscore', 'doge_rsi',
            'tsla_ret_1h', 'tsla_market_open', 'tsla_vol_zscore'
        ]
        for feat in market_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Opcionales
        for feat in ['mentions_tesla', 'mentions_doge']:
            if feat in df.columns:
                feature_cols.append(feat)
        
        X = df[feature_cols].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if is_train:
            self.feature_names = X.columns.tolist()
        
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y = df['impact_class'].values if 'impact_class' in df.columns else None
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena clasificadores"""
        print("\n" + "="*70)
        print("🎯 ENTRENAMIENTO FINAL - IMPACT CLASSIFIER")
        print("="*70)
        
        df = self.create_impact_labels(df)
        X, y = self.prepare_features(df, is_train=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Random Forest
        print("\n📊 [1/3] Entrenando Random Forest...")
        rf_scores = []
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            random_state=settings.RANDOM_SEED,
            n_jobs=-1
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['random_forest'].fit(X[train_idx], y[train_idx])
            pred = self.models['random_forest'].predict(X[val_idx])
            accuracy = (pred == y[val_idx]).mean()
            rf_scores.append(accuracy)
        
        self.models['random_forest'].fit(X, y)
        print(f"   ✅ Random Forest: Accuracy = {np.mean(rf_scores):.4f} (±{np.std(rf_scores):.4f})")
        
        # XGBoost
        print("\n📊 [2/3] Entrenando XGBoost Classifier...")
        xgb_scores = []
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=settings.RANDOM_SEED,
            tree_method='hist'
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['xgboost'].fit(X[train_idx], y[train_idx])
            pred = self.models['xgboost'].predict(X[val_idx])
            accuracy = (pred == y[val_idx]).mean()
            xgb_scores.append(accuracy)
        
        self.models['xgboost'].fit(X, y)
        print(f"   ✅ XGBoost: Accuracy = {np.mean(xgb_scores):.4f} (±{np.std(xgb_scores):.4f})")
        
        # LightGBM
        print("\n📊 [3/3] Entrenando LightGBM Classifier...")
        lgb_scores = []
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=settings.RANDOM_SEED,
            verbose=-1
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['lightgbm'].fit(X[train_idx], y[train_idx])
            pred = self.models['lightgbm'].predict(X[val_idx])
            accuracy = (pred == y[val_idx]).mean()
            lgb_scores.append(accuracy)
        
        self.models['lightgbm'].fit(X, y)
        print(f"   ✅ LightGBM: Accuracy = {np.mean(lgb_scores):.4f} (±{np.std(lgb_scores):.4f})")
        
        # Guardar métricas
        self.metrics = {
            'random_forest': {'accuracy': np.mean(rf_scores), 'std': np.std(rf_scores)},
            'xgboost': {'accuracy': np.mean(xgb_scores), 'std': np.std(xgb_scores)},
            'lightgbm': {'accuracy': np.mean(lgb_scores), 'std': np.std(lgb_scores)}
        }
        
        self.is_trained = True
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO IMPACT CLASSIFIER FINAL COMPLETADO")
        print("="*70)
    
    def predict(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Predice clase"""
        X, _ = self.prepare_features(df, is_train=False)
        return self.models[model_name].predict(X)
    
    def predict_proba(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Predice probabilidades"""
        X, _ = self.prepare_features(df, is_train=False)
        return self.models[model_name].predict_proba(X)
    
    def get_classification_metrics(self, df: pd.DataFrame, model_name: str = 'xgboost') -> Dict:
        """Métricas detalladas"""
        df = self.create_impact_labels(df)
        X, y_true = self.prepare_features(df, is_train=False)
        
        y_pred = self.models[model_name].predict(X)
        y_proba = self.models[model_name].predict_proba(X)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        class_names = ['No Impact', 'DOGE Only', 'TSLA Only', 'Both']
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names,
            'support': support
        }