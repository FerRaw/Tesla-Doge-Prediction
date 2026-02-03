"""
Modelos predictivos para DOGE, TSLA y clasificador de impacto
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from src.models.base_predictor import BasePredictor


class DOGEPredictor(BasePredictor):
    """Predictor de retornos de DOGECOIN"""
    
    def __init__(self, version: str = "v1"):
        super().__init__("doge_predictor", version)
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features para DOGE"""
        
        # Features base de mercado
        feature_cols = [
            'doge_ret_1h', 'doge_vol_zscore', 'doge_buy_pressure', 'doge_rsi',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Features de sentimiento con lags (optimizados por Granger: 1h, 2h, 3h)
        sentiment_features = [
            'sentiment_ensemble', 'relevance_score',
            'sentiment_ensemble_lag1', 'sentiment_ensemble_lag2', 'sentiment_ensemble_lag3',
            'relevance_score_lag1', 'relevance_score_lag2', 'relevance_score_lag3'
        ]
        
        # Añadir features disponibles
        for feat in sentiment_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Guardar nombres de features (solo en entrenamiento)
        if is_train:
            self.feature_names = feature_cols
        
        # Extraer features
        X = df[feature_cols].copy()
        
        # Target (solo si es entrenamiento)
        y = df['TARGET_DOGE'].values if is_train and 'TARGET_DOGE' in df.columns else None
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """
        Entrena todos los modelos con Time Series CV
        
        Args:
            df: DataFrame con features y target
            n_splits: Número de splits para CV
        """
        print("\n" + "="*70)
        print(f"🐕 ENTRENANDO {self.model_name.upper()}")
        print("="*70)
        
        X, y = self.prepare_features(df, is_train=True)
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=settings.TEST_SIZE_HOURS)
        
        # ==================================================================
        # 1. XGBoost
        # ==================================================================
        print("\n📊 [1/5] Entrenando XGBoost...")
        xgb_scores = []
        
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=settings.XGB_N_ESTIMATORS,
            learning_rate=settings.XGB_LEARNING_RATE,
            max_depth=settings.XGB_MAX_DEPTH,
            min_child_weight=settings.XGB_MIN_CHILD_WEIGHT,
            subsample=settings.XGB_SUBSAMPLE,
            colsample_bytree=settings.XGB_COLSAMPLE_BYTREE,
            random_state=settings.RANDOM_SEED,
            objective='reg:squarederror',
            verbosity=0
        )
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.models['xgboost'].fit(X_train, y_train, verbose=False)
            pred = self.models['xgboost'].predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            xgb_scores.append(rmse)
        
        self.metrics['xgboost'] = {
            'cv_rmse_mean': np.mean(xgb_scores),
            'cv_rmse_std': np.std(xgb_scores)
        }
        
        # Entrenar en todo el dataset
        self.models['xgboost'].fit(X, y)
        self._print_cv_results('xgboost', xgb_scores)
        
        # ==================================================================
        # 2. LightGBM
        # ==================================================================
        print("\n📊 [2/5] Entrenando LightGBM...")
        lgb_scores = []
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=settings.LGB_N_ESTIMATORS,
            learning_rate=settings.LGB_LEARNING_RATE,
            max_depth=settings.LGB_MAX_DEPTH,
            num_leaves=settings.LGB_NUM_LEAVES,
            random_state=settings.RANDOM_SEED,
            verbose=-1
        )
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.models['lightgbm'].fit(X_train, y_train)
            pred = self.models['lightgbm'].predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            lgb_scores.append(rmse)
        
        self.metrics['lightgbm'] = {
            'cv_rmse_mean': np.mean(lgb_scores),
            'cv_rmse_std': np.std(lgb_scores)
        }
        
        self.models['lightgbm'].fit(X, y)
        self._print_cv_results('lightgbm', lgb_scores)
        
        # ==================================================================
        # 3. Elastic Net
        # ==================================================================
        print("\n📊 [3/5] Entrenando Elastic Net...")
        en_scores = []
        
        self.models['elastic_net'] = ElasticNet(
            alpha=settings.EN_ALPHA,
            l1_ratio=settings.EN_L1_RATIO,
            random_state=settings.RANDOM_SEED,
            max_iter=1000
        )
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.models['elastic_net'].fit(X_train, y_train)
            pred = self.models['elastic_net'].predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            en_scores.append(rmse)
        
        self.metrics['elastic_net'] = {
            'cv_rmse_mean': np.mean(en_scores),
            'cv_rmse_std': np.std(en_scores)
        }
        
        self.models['elastic_net'].fit(X, y)
        self._print_cv_results('elastic_net', en_scores)
        
        # ==================================================================
        # 4. LSTM
        # ==================================================================
        print("\n📊 [4/5] Entrenando LSTM...")
        lstm_scores = []
        
        X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
        
        for train_idx, val_idx in tscv.split(X):
            X_train = X_lstm[train_idx]
            X_val = X_lstm[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = keras.Sequential([
                layers.LSTM(settings.LSTM_UNITS, activation='relu', 
                           input_shape=(1, X.shape[1])),
                layers.Dropout(settings.LSTM_DROPOUT),
                layers.Dense(32, activation='relu'),
                layers.Dropout(settings.LSTM_DROPOUT),
                layers.Dense(1)
            ])
            
            model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            model.fit(X_train, y_train, 
                     epochs=settings.LSTM_EPOCHS, 
                     batch_size=settings.LSTM_BATCH_SIZE,
                     verbose=0, validation_data=(X_val, y_val))
            
            pred = model.predict(X_val, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            lstm_scores.append(rmse)
        
        self.metrics['lstm'] = {
            'cv_rmse_mean': np.mean(lstm_scores),
            'cv_rmse_std': np.std(lstm_scores)
        }
        
        # Entrenar modelo final
        final_lstm = keras.Sequential([
            layers.LSTM(settings.LSTM_UNITS, activation='relu', 
                       input_shape=(1, X.shape[1])),
            layers.Dropout(settings.LSTM_DROPOUT),
            layers.Dense(32, activation='relu'),
            layers.Dropout(settings.LSTM_DROPOUT),
            layers.Dense(1)
        ])
        final_lstm.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
        final_lstm.fit(X_lstm, y, 
                      epochs=settings.LSTM_EPOCHS, 
                      batch_size=settings.LSTM_BATCH_SIZE, 
                      verbose=0)
        
        self.models['lstm'] = final_lstm
        self._print_cv_results('lstm', lstm_scores)
        
        # ==================================================================
        # 5. Ensemble
        # ==================================================================
        print("\n📊 [5/5] Creando Ensemble...")
        self._create_ensemble_weights()
        
        # Actualizar metadata
        self.is_trained = True
        self.update_metadata(
            trained_at=pd.Timestamp.now().isoformat(),
            training_samples=len(df),
            features_count=len(self.feature_names),
            cv_splits=n_splits
        )
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO DOGE COMPLETADO")
        print("="*70)
    
    def predict(self, df: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """Predice retornos de DOGE"""
        X, _ = self.prepare_features(df, is_train=False)
        
        if model_name == 'ensemble':
            predictions = {}
            
            # Modelos tree-based y linear
            for name in ['xgboost', 'lightgbm', 'elastic_net']:
                predictions[name] = self.models[name].predict(X)
            
            # LSTM
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            predictions['lstm'] = self.models['lstm'].predict(X_lstm, verbose=0).flatten()
            
            # Combinar con pesos
            ensemble_pred = np.zeros(len(X))
            for name, weight in self.ensemble_weights.items():
                ensemble_pred += weight * predictions[name]
            
            return ensemble_pred
        else:
            if model_name == 'lstm':
                X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
                return self.models[model_name].predict(X_lstm, verbose=0).flatten()
            return self.models[model_name].predict(X)


class TSLAPredictor(BasePredictor):
    """Predictor de retornos de TESLA"""
    
    def __init__(self, version: str = "v1"):
        super().__init__("tsla_predictor", version)
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features para TSLA"""
        
        # Features base
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
        
        if is_train:
            self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        y = df['TARGET_TSLA'].values if is_train and 'TARGET_TSLA' in df.columns else None
        
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena modelos para TSLA (misma estructura que DOGE)"""
        print("\n" + "="*70)
        print(f"🚗 ENTRENANDO {self.model_name.upper()}")
        print("="*70)
        
        X, y = self.prepare_features(df, is_train=True)
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=settings.TEST_SIZE_HOURS)
        
        # XGBoost
        print("\n📊 [1/4] Entrenando XGBoost...")
        xgb_scores = []
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=settings.RANDOM_SEED, verbosity=0
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['xgboost'].fit(X[train_idx], y[train_idx])
            pred = self.models['xgboost'].predict(X[val_idx])
            xgb_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['xgboost'] = {
            'cv_rmse_mean': np.mean(xgb_scores),
            'cv_rmse_std': np.std(xgb_scores)
        }
        self.models['xgboost'].fit(X, y)
        self._print_cv_results('xgboost', xgb_scores)
        
        # LightGBM
        print("\n📊 [2/4] Entrenando LightGBM...")
        lgb_scores = []
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=settings.RANDOM_SEED, verbose=-1
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['lightgbm'].fit(X[train_idx], y[train_idx])
            pred = self.models['lightgbm'].predict(X[val_idx])
            lgb_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['lightgbm'] = {
            'cv_rmse_mean': np.mean(lgb_scores),
            'cv_rmse_std': np.std(lgb_scores)
        }
        self.models['lightgbm'].fit(X, y)
        self._print_cv_results('lightgbm', lgb_scores)
        
        # Elastic Net
        print("\n📊 [3/4] Entrenando Elastic Net...")
        en_scores = []
        self.models['elastic_net'] = ElasticNet(
            alpha=0.01, l1_ratio=0.5, random_state=settings.RANDOM_SEED
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['elastic_net'].fit(X[train_idx], y[train_idx])
            pred = self.models['elastic_net'].predict(X[val_idx])
            en_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['elastic_net'] = {
            'cv_rmse_mean': np.mean(en_scores),
            'cv_rmse_std': np.std(en_scores)
        }
        self.models['elastic_net'].fit(X, y)
        self._print_cv_results('elastic_net', en_scores)
        
        # Ensemble
        print("\n📊 [4/4] Creando Ensemble...")
        self._create_ensemble_weights()
        
        self.is_trained = True
        self.update_metadata(
            trained_at=pd.Timestamp.now().isoformat(),
            training_samples=len(df),
            features_count=len(self.feature_names),
            cv_splits=n_splits
        )
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO TSLA COMPLETADO")
        print("="*70)
    
    def predict(self, df: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """Predice retornos de TSLA"""
        X, _ = self.prepare_features(df, is_train=False)
        
        if model_name == 'ensemble':
            predictions = {}
            for name in ['xgboost', 'lightgbm', 'elastic_net']:
                predictions[name] = self.models[name].predict(X)
            
            ensemble_pred = np.zeros(len(X))
            for name, weight in self.ensemble_weights.items():
                ensemble_pred += weight * predictions[name]
            
            return ensemble_pred
        else:
            return self.models[model_name].predict(X)


class ImpactClassifier(BasePredictor):
    """Clasificador de impacto de tweets"""
    
    def __init__(self, version: str = "v1"):
        super().__init__("impact_classifier", version)
        self.impact_threshold = 0.02  # 2% movimiento
    
    def create_impact_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels de impacto"""
        df = df.copy()
        
        doge_impact = abs(df['TARGET_DOGE']) > self.impact_threshold
        tsla_impact = abs(df['TARGET_TSLA']) > self.impact_threshold
        
        df['impact_class'] = 0
        df.loc[doge_impact & ~tsla_impact, 'impact_class'] = 1
        df.loc[~doge_impact & tsla_impact, 'impact_class'] = 2
        df.loc[doge_impact & tsla_impact, 'impact_class'] = 3
        
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features para clasificador"""
        
        feature_cols = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'doge_vol_zscore', 'tsla_vol_zscore', 'tsla_market_open'
        ]
        
        # Añadir sentiment/relevance
        for feat in ['sentiment_ensemble', 'relevance_score']:
            if feat in df.columns:
                feature_cols.append(feat)
        
        if is_train:
            self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        y = df['impact_class'].values if is_train and 'impact_class' in df.columns else None
        
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame):
        """Entrena clasificador"""
        print("\n" + "="*70)
        print(f"🎯 ENTRENANDO {self.model_name.upper()}")
        print("="*70)
        
        df = self.create_impact_labels(df)
        X, y = self.prepare_features(df, is_train=True)
        
        # Random Forest
        print("\n📊 [1/2] Entrenando Random Forest...")
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=settings.RANDOM_SEED,
            class_weight='balanced'
        )
        self.models['random_forest'].fit(X, y)
        
        # XGBoost
        print("📊 [2/2] Entrenando XGBoost Classifier...")
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=settings.RANDOM_SEED,
            eval_metric='mlogloss',
            verbosity=0
        )
        self.models['xgboost'].fit(X, y)
        
        self.is_trained = True
        self.update_metadata(
            trained_at=pd.Timestamp.now().isoformat(),
            training_samples=len(df)
        )
        
        print("\n✅ ENTRENAMIENTO IMPACT CLASSIFIER COMPLETADO")
    
    def predict(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Predice clase de impacto"""
        X, _ = self.prepare_features(df, is_train=False)
        return self.models[model_name].predict(X)
    
    def predict_proba(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Predice probabilidades"""
        X, _ = self.prepare_features(df, is_train=False)
        return self.models[model_name].predict_proba(X)