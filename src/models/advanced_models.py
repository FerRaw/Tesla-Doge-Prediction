"""
Modelos avanzados: Bi-LSTM con Attention, TCN, Stacking Ensemble
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras  # Importamos el Keras 3 independiente al final
from keras import layers, Model  # Importamos desde Keras, no desde TF


@keras.saving.register_keras_serializable(package="MyModels")
class AttentionLayer(layers.Layer):
    """Mecanismo de atención para secuencias temporales"""
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        super().build(input_shape)
    
    def call(self, x):
        # x: (batch, timesteps, features)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, axis=-1)
        weighted = x * ait
        output = tf.reduce_sum(weighted, axis=1)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def create_bilstm_attention_model(
    input_shape: tuple,
    lstm_units: int = 128,
    dropout: float = 0.3,
    attention_units: int = 64
) -> Model:
    """
    Bi-LSTM con mecanismo de atención
    
    Captura dependencias bidireccionales y enfoca en timesteps relevantes
    """
    inputs = keras.Input(shape=input_shape)
    
    # Bi-LSTM layers
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
    )(inputs)
    
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout)
    )(x)
    
    # Attention mechanism
    x = AttentionLayer(attention_units)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Attention')
    
    # Compilar con directional_loss
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=directional_loss,
        metrics=['mae']
    )
    
    return model


def create_tcn_model(
    input_shape: tuple,
    num_filters: int = 64,
    kernel_size: int = 3,
    dilations: list = [1, 2, 4, 8],
    dropout: float = 0.2
) -> Model:
    """
    Temporal Convolutional Network (TCN)
    
    Captura patrones temporales con convoluciones causales dilatadas
    Más eficiente que LSTM para secuencias largas
    """
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    
    # Stacked dilated causal convolutions
    for dilation_rate in dilations:
        # Residual block
        residual = x
        
        # Causal conv
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(x)
        
        x = layers.Dropout(dropout)(x)
        
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(x)
        
        x = layers.Dropout(dropout)(x)
        
        # Residual connection (match dimensions if needed)
        if residual.shape[-1] != num_filters:
            residual = layers.Conv1D(num_filters, 1, padding='same')(residual)
        
        x = layers.Add()([x, residual])
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TCN')
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_transformer_model(
    input_shape: tuple,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_transformer_blocks: int = 2,
    dropout: float = 0.2
) -> Model:
    """
    Transformer encoder para series temporales
    
    Captura dependencias de largo alcance mejor que LSTM
    """
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    
    # Positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embeddings = layers.Embedding(
        input_dim=input_shape[0],
        output_dim=input_shape[1]
    )(positions)
    
    x = x + position_embeddings
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[1] // num_heads,
            dropout=dropout
        )(x, x)
        
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(input_shape[1])
        ])
        
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Transformer')
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_hybrid_cnn_lstm_model(
    input_shape: tuple,
    cnn_filters: int = 64,
    lstm_units: int = 64,
    dropout: float = 0.3
) -> Model:
    """
    Híbrido CNN-LSTM
    
    CNN extrae features locales, LSTM captura dependencias temporales
    """
    inputs = keras.Input(shape=input_shape)
    
    # CNN layers (extract local patterns)
    x = layers.Conv1D(cnn_filters, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Conv1D(cnn_filters * 2, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    
    # LSTM layers (capture temporal dependencies)
    x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(x)
    x = layers.LSTM(lstm_units // 2, dropout=dropout)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Hybrid')
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# =============================================================================
# Callbacks avanzados
# =============================================================================

def get_advanced_callbacks(model_name: str, patience: int = 15):
    """
    Callbacks para training avanzado
    """
    from pathlib import Path
    
    # Crear directorio de checkpoints si no existe
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        ),
        
        # Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-6,
            verbose=0
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    return callbacks


# =============================================================================
# Funciones de pérdida personalizadas
# =============================================================================

def directional_loss(y_true, y_pred):
    """
    Pérdida que penaliza errores direccionales
    
    Más importante predecir la dirección correcta que la magnitud exacta
    """
    # MSE base
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Directional penalty
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    
    directional_error = tf.reduce_mean(
        tf.abs(direction_true - direction_pred)
    )
    
    # Combinar (70% MSE, 30% dirección)
    return 0.7 * mse + 0.3 * directional_error


def quantile_loss(quantile=0.5):
    """
    Quantile loss para modelar incertidumbre
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(
            tf.maximum(quantile * error, (quantile - 1) * error)
        )
    return loss