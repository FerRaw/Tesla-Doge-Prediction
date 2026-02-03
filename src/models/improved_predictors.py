"""
Predictores mejorados con modelos avanzados y ensemble sofisticado
"""
import pandas as pd
import numpy as np
import keras
from typing import Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import (
    StackingRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from src.models.base_predictor import BasePredictor
from src.models.advanced_models import (
    create_bilstm_attention_model,
    create_tcn_model,
    create_transformer_model,
    create_hybrid_cnn_lstm_model,
    get_advanced_callbacks,
    directional_loss
)


class ImprovedDOGEPredictor(BasePredictor):
    """Predictor DOGE mejorado con arquitecturas avanzadas"""
    
    def __init__(self, version: str = "v2", use_advanced_models: bool = True):
        super().__init__("doge_predictor_improved", version)
        self.use_advanced_models = use_advanced_models
        self.lookback = 24  # Usar últimas 24 horas como secuencia
    
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
            'vol_regime', 'atr', 'session_', 'is_weekend'
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
        y = df['TARGET_DOGE'].values if is_train and 'TARGET_DOGE' in df.columns else None
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray = None, lookback: int = None):
        """
        Prepara secuencias temporales para modelos recurrentes
        
        Transforma de (samples, features) a (samples, lookback, features)
        """
        if lookback is None:
            lookback = self.lookback
        
        X_seq = []
        y_seq = []
        
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena con modelos avanzados"""
        print("\n" + "="*70)
        print(f"🚀 ENTRENANDO {self.model_name.upper()} (VERSIÓN MEJORADA)")
        print("="*70)
        
        X, y = self.prepare_features(df, is_train=True)
        
        # Time Series CV
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=settings.TEST_SIZE_HOURS)
        
        # =================================================================
        # 1. XGBoost (con más árboles y profundidad)
        # =================================================================
        print("\n📊 [1/8] Entrenando XGBoost Mejorado...")
        xgb_scores = []
        
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=500,  # Aumentado
            learning_rate=0.01,  # Reducido para mejor generalización
            max_depth=8,  # Aumentado
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=settings.RANDOM_SEED,
            verbosity=0,
            tree_method='hist'  # Más rápido
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
            'cv_rmse_std': np.std(xgb_scores)
        }
        self.models['xgboost'].fit(X, y)
        self._print_cv_results('xgboost', xgb_scores)
        
        # =================================================================
        # 2. LightGBM (optimizado)
        # =================================================================
        print("\n📊 [2/8] Entrenando LightGBM Mejorado...")
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
            self.models['lightgbm'].fit(X[train_idx], y[train_idx])
            pred = self.models['lightgbm'].predict(X[val_idx])
            lgb_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['lightgbm'] = {
            'cv_rmse_mean': np.mean(lgb_scores),
            'cv_rmse_std': np.std(lgb_scores)
        }
        self.models['lightgbm'].fit(X, y)
        self._print_cv_results('lightgbm', lgb_scores)
        
        # =================================================================
        # 3. CatBoost (nuevo - muy potente)
        # =================================================================
        print("\n📊 [3/8] Entrenando CatBoost...")
        cat_scores = []
        
        self.models['catboost'] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=settings.RANDOM_SEED,
            verbose=False
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['catboost'].fit(X[train_idx], y[train_idx])
            pred = self.models['catboost'].predict(X[val_idx])
            cat_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.metrics['catboost'] = {
            'cv_rmse_mean': np.mean(cat_scores),
            'cv_rmse_std': np.std(cat_scores)
        }
        self.models['catboost'].fit(X, y)
        self._print_cv_results('catboost', cat_scores)
        
        if not self.use_advanced_models:
            # Solo modelos tree-based
            print("\n📊 Creando Ensemble...")
            self._create_ensemble_weights()
            self.is_trained = True
            return
        
        # =================================================================
        # 4. Bi-LSTM con Attention
        # =================================================================
        print("\n📊 [4/8] Entrenando Bi-LSTM con Attention...")
        X_seq, y_seq = self.prepare_sequences(X, y, self.lookback)
        
        bilstm_scores = []
        input_shape = (self.lookback, X.shape[1])
        
        for train_idx, val_idx in tscv.split(X_seq):
            model = create_bilstm_attention_model(
                input_shape,
                lstm_units=128,
                dropout=0.3,
                attention_units=64
            )
            
            model.compile(
                optimizer=keras.optimizers.Adam(0.001),
                loss=directional_loss
            )
            
            callbacks = get_advanced_callbacks('bilstm_attention', patience=20)
            
            model.fit(
                X_seq[train_idx], y_seq[train_idx],
                validation_data=(X_seq[val_idx], y_seq[val_idx]),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            pred = model.predict(X_seq[val_idx], verbose=0)
            bilstm_scores.append(np.sqrt(mean_squared_error(y_seq[val_idx], pred)))
        
        self.metrics['bilstm_attention'] = {
            'cv_rmse_mean': np.mean(bilstm_scores),
            'cv_rmse_std': np.std(bilstm_scores)
        }
        
        # Entrenar en todo el dataset
        final_bilstm = create_bilstm_attention_model(input_shape, 128, 0.3, 64)
        final_bilstm.compile(optimizer=keras.optimizers.Adam(0.001), loss=directional_loss)
        final_bilstm.fit(X_seq, y_seq, epochs=100, batch_size=64, verbose=0)
        self.models['bilstm_attention'] = final_bilstm
        self._print_cv_results('bilstm_attention', bilstm_scores)
        
        # =================================================================
        # 5. TCN (Temporal Convolutional Network)
        # =================================================================
        print("\n📊 [5/8] Entrenando TCN...")
        tcn_scores = []
        
        for train_idx, val_idx in tscv.split(X_seq):
            model = create_tcn_model(
                input_shape,
                num_filters=64,
                kernel_size=3,
                dilations=[1, 2, 4, 8],
                dropout=0.2
            )
            
            model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            callbacks = get_advanced_callbacks('tcn', patience=20)
            
            model.fit(
                X_seq[train_idx], y_seq[train_idx],
                validation_data=(X_seq[val_idx], y_seq[val_idx]),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            pred = model.predict(X_seq[val_idx], verbose=0)
            tcn_scores.append(np.sqrt(mean_squared_error(y_seq[val_idx], pred)))
        
        self.metrics['tcn'] = {
            'cv_rmse_mean': np.mean(tcn_scores),
            'cv_rmse_std': np.std(tcn_scores)
        }
        
        final_tcn = create_tcn_model(input_shape, 64, 3, [1, 2, 4, 8], 0.2)
        final_tcn.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
        final_tcn.fit(X_seq, y_seq, epochs=100, batch_size=64, verbose=0)
        self.models['tcn'] = final_tcn
        self._print_cv_results('tcn', tcn_scores)
        
        # =================================================================
        # 6. Transformer
        # =================================================================
        print("\n📊 [6/8] Entrenando Transformer...")
        transformer_scores = []
        
        for train_idx, val_idx in tscv.split(X_seq):
            model = create_transformer_model(
                input_shape,
                num_heads=4,
                ff_dim=128,
                num_transformer_blocks=2,
                dropout=0.2
            )
            
            model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            callbacks = get_advanced_callbacks('transformer', patience=20)
            
            model.fit(
                X_seq[train_idx], y_seq[train_idx],
                validation_data=(X_seq[val_idx], y_seq[val_idx]),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            pred = model.predict(X_seq[val_idx], verbose=0)
            transformer_scores.append(np.sqrt(mean_squared_error(y_seq[val_idx], pred)))
        
        self.metrics['transformer'] = {
            'cv_rmse_mean': np.mean(transformer_scores),
            'cv_rmse_std': np.std(transformer_scores)
        }
        
        final_transformer = create_transformer_model(input_shape, 4, 128, 2, 0.2)
        final_transformer.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
        final_transformer.fit(X_seq, y_seq, epochs=100, batch_size=64, verbose=0)
        self.models['transformer'] = final_transformer
        self._print_cv_results('transformer', transformer_scores)
        
        # =================================================================
        # 7. CNN-LSTM Hybrid
        # =================================================================
        print("\n📊 [7/8] Entrenando CNN-LSTM Hybrid...")
        hybrid_scores = []
        
        for train_idx, val_idx in tscv.split(X_seq):
            model = create_hybrid_cnn_lstm_model(input_shape, 64, 64, 0.3)
            model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            callbacks = get_advanced_callbacks('cnn_lstm', patience=20)
            
            model.fit(
                X_seq[train_idx], y_seq[train_idx],
                validation_data=(X_seq[val_idx], y_seq[val_idx]),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            pred = model.predict(X_seq[val_idx], verbose=0)
            hybrid_scores.append(np.sqrt(mean_squared_error(y_seq[val_idx], pred)))
        
        self.metrics['cnn_lstm'] = {
            'cv_rmse_mean': np.mean(hybrid_scores),
            'cv_rmse_std': np.std(hybrid_scores)
        }
        
        final_hybrid = create_hybrid_cnn_lstm_model(input_shape, 64, 64, 0.3)
        final_hybrid.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
        final_hybrid.fit(X_seq, y_seq, epochs=100, batch_size=64, verbose=0)
        self.models['cnn_lstm'] = final_hybrid
        self._print_cv_results('cnn_lstm', hybrid_scores)
        
        # =================================================================
        # 8. Stacking Ensemble
        # =================================================================
        print("\n📊 [8/8] Creando Stacking Ensemble...")
        self._create_stacking_ensemble(X, y, tscv)
        
        self.is_trained = True
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO MEJORADO COMPLETADO")
        print("="*70)
    
    def _create_stacking_ensemble(self, X, y, tscv):
        """Crea ensemble con stacking (meta-learner)"""
        base_models = [
            ('xgb', self.models['xgboost']),
            ('lgb', self.models['lightgbm']),
            ('cat', self.models['catboost'])
        ]
        
        # Meta-learner (Gradient Boosting)
        meta_learner = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=settings.RANDOM_SEED
        )
        
        # CORRECCIÓN: usar passthrough=False y quitar cv del StackingRegressor
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            passthrough=False,
            n_jobs=-1
        )
        
        # Entrenar el stacking con validación cruzada manual
        stacking_scores = []
        for train_idx, val_idx in tscv.split(X):
            stacking.fit(X[train_idx], y[train_idx])
            pred = stacking.predict(X[val_idx])
            stacking_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        # Entrenar en todos los datos
        stacking.fit(X, y)
        self.models['stacking'] = stacking
        
        self.metrics['stacking'] = {
            'cv_rmse_mean': np.mean(stacking_scores),
            'cv_rmse_std': np.std(stacking_scores)
        }
        
        print(f"   ✅ Stacking Ensemble creado - RMSE: {np.mean(stacking_scores):.6f} (±{np.std(stacking_scores):.6f})")
    
    def predict(self, df: pd.DataFrame, model_name: str = 'stacking') -> np.ndarray:
        """Predice con modelo especificado"""
        X, _ = self.prepare_features(df, is_train=False)
        
        # Modelos recurrentes necesitan secuencias
        if model_name in ['bilstm_attention', 'tcn', 'transformer', 'cnn_lstm']:
            X_seq, _ = self.prepare_sequences(X, lookback=self.lookback)
            return self.models[model_name].predict(X_seq, verbose=0).flatten()
        else:
            return self.models[model_name].predict(X)


# =================================================================
# IMPROVED TSLA PREDICTOR
# =================================================================
class ImprovedTSLAPredictor(ImprovedDOGEPredictor):
    """
    Predictor TSLA mejorado (hereda de DOGE)
    Misma arquitectura pero para acciones de Tesla
    """
    
    def __init__(self, version: str = "v2", use_advanced_models: bool = True):
        BasePredictor.__init__(self, "tsla_predictor_improved", version)
        self.use_advanced_models = use_advanced_models
        self.lookback = 24
        self.scaler = StandardScaler()
        self.ensemble_weights = {}
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features para TSLA (features correctas del proyecto)"""
        
        # Features base de TSLA
        feature_cols = [
            'tsla_ret_1h', 'tsla_market_open', 'tsla_vol_zscore',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Añadir features de sentimiento con lag óptimo (1h según Granger)
        sentiment_features = [
            'sentiment_ensemble', 'relevance_score',
            'sentiment_ensemble_lag1', 'relevance_score_lag1'
        ]
        
        # Solo añadir si existen en el DataFrame
        for feat in sentiment_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        # Features adicionales opcionales de momentum
        optional_features = [
            'tsla_momentum_3h', 'tsla_momentum_6h', 'tsla_momentum_12h'
        ]
        
        for feat in optional_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        X = df[feature_cols].copy()
        
        # Wavelets solo en retornos (si hay suficientes datos)
        if 'tsla_ret_1h' in X.columns and len(X) >= 32:
            try:
                ret_wavelet = self._apply_wavelets(X['tsla_ret_1h'].values)
                X['tsla_ret_wavelet_approx'] = ret_wavelet[0]
                X['tsla_ret_wavelet_detail'] = ret_wavelet[1]
            except:
                X['tsla_ret_wavelet_approx'] = 0
                X['tsla_ret_wavelet_detail'] = 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if is_train:
            self.feature_names = X.columns.tolist()
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Target
        y = df['TARGET_TSLA'].values if 'TARGET_TSLA' in df.columns else None
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena TSLA usando el método del padre con logs personalizados"""
        print("\n" + "="*70)
        print("🚀 ENTRENAMIENTO MEJORADO - TSLA PREDICTOR")
        print("="*70)
        
        # Llamar al método train del padre que ya tiene toda la lógica
        # Esto usará prepare_features de TSLA (sobrescrito arriba)
        super().train(df, n_splits)
        
        # Override del mensaje final
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO TSLA COMPLETADO")
        print("="*70)

# =================================================================
# IMPACT CLASSIFIER
# =================================================================
class ImpactClassifier(BasePredictor):
    """
    Clasificador de impacto de tweets en mercados
    
    Clases:
    0: Sin impacto
    1: Solo DOGE
    2: Solo TSLA
    3: Ambos (DOGE + TSLA)
    """
    
    def __init__(self, version: str = "v1"):
        super().__init__("impact_classifier", version)
        self.impact_threshold = 0.02  # 2% movimiento
        self.scaler = StandardScaler()
    
    def create_impact_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels de impacto basados en movimientos"""
        df = df.copy()
        
        doge_impact = abs(df['TARGET_DOGE']) > self.impact_threshold
        tsla_impact = abs(df['TARGET_TSLA']) > self.impact_threshold
        
        # Clase 0: Sin impacto
        df['impact_class'] = 0
        
        # Clase 1: Solo DOGE
        df.loc[doge_impact & ~tsla_impact, 'impact_class'] = 1
        
        # Clase 2: Solo TSLA
        df.loc[~doge_impact & tsla_impact, 'impact_class'] = 2
        
        # Clase 3: Ambos
        df.loc[doge_impact & tsla_impact, 'impact_class'] = 3
        
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features para clasificación"""
        
        feature_cols = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'tweet_sentiment', 'tweet_volume', 'tweet_engagement',
            'sentiment_volatility'
        ]
        
        # Añadir contexto de mercado
        market_features = [
            'doge_ret_1h', 'doge_vol_zscore', 'doge_rsi',
            'tsla_ret_1h', 'tsla_vol_zscore', 'tsla_rsi'
        ]
        
        available_features = [f for f in feature_cols + market_features if f in df.columns]
        
        X = df[available_features].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if is_train:
            self.feature_names = X.columns.tolist()
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Target
        y = df['impact_class'].values if 'impact_class' in df.columns else None
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena clasificador de impacto"""
        print("\n" + "="*70)
        print("🎯 ENTRENAMIENTO - IMPACT CLASSIFIER")
        print("="*70)
        
        # Crear labels
        df = self.create_impact_labels(df)
        
        X, y = self.prepare_features(df, is_train=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Random Forest Classifier
        print("\n📊 [1/3] Entrenando Random Forest...")
        rf_scores = []
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
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
        print(f"   ✅ Random Forest: Accuracy = {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
        
        # XGBoost Classifier
        print("\n📊 [2/3] Entrenando XGBoost Classifier...")
        xgb_scores = []
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=settings.RANDOM_SEED,
            tree_method='hist'
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['xgboost'].fit(X[train_idx], y[train_idx])
            pred = self.models['xgboost'].predict(X[val_idx])
            accuracy = (pred == y[val_idx]).mean()
            xgb_scores.append(accuracy)
        
        self.models['xgboost'].fit(X, y)
        print(f"   ✅ XGBoost: Accuracy = {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")
        
        # LightGBM Classifier
        print("\n📊 [3/3] Entrenando LightGBM Classifier...")
        lgb_scores = []
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=settings.RANDOM_SEED,
            verbose=-1
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['lightgbm'].fit(X[train_idx], y[train_idx])
            pred = self.models['lightgbm'].predict(X[val_idx])
            accuracy = (pred == y[val_idx]).mean()
            lgb_scores.append(accuracy)
        
        self.models['lightgbm'].fit(X, y)
        print(f"   ✅ LightGBM: Accuracy = {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}")
        
        self.is_trained = True
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO IMPACT CLASSIFIER COMPLETADO")
        print("="*70)
    
    def predict(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Predice clase de impacto"""
        X, _ = self.prepare_features(df, is_train=False)
        return self.models[model_name].predict(X)
    
    def predict_proba(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Predice probabilidades de clase"""
        X, _ = self.prepare_features(df, is_train=False)
        return self.models[model_name].predict_proba(X)
