"""
╔══════════════════════════════════════════════════════════════════════╗
║       XSIVE AUDIO CLASSIFIER — FASE 3: ARQUITECTURA CNN             ║
║        Deep Convolutional Neural Network para Clasificación          ║
║                    Binaria de Audio Espectral                        ║
╚══════════════════════════════════════════════════════════════════════╝

Diseño de la CNN profunda optimizada para clasificación espectral:

  Input: Mel Spectrogram como imagen monocanal (128×128×1)

  Bloque 1 → Conv2D(32)  → BatchNorm → ReLU → MaxPool → Dropout
  Bloque 2 → Conv2D(64)  → BatchNorm → ReLU → MaxPool → Dropout
  Bloque 3 → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout
  Bloque 4 → Conv2D(256) → BatchNorm → ReLU → GlobalAvgPool

  Head     → Dense(256) → Dense(64) → Dense(1, Sigmoid)
  Output   → Probabilidad [0.0, 1.0] de ser Clase 1 (Aprobado)
"""

import logging
import os
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

from config import (
    INPUT_SHAPE, CONV_FILTERS, KERNEL_SIZE, POOL_SIZE,
    DROPOUT_RATE, DENSE_UNITS, LEARNING_RATE,
    MODELS_DIR, MODEL_FILENAME, WEIGHTS_FILENAME
)

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("XsiveCNN")

# Forzar determinismo en operaciones de TensorFlow
tf.random.set_seed(42)


# ─────────────────────────────────────────────────────────────────────
# BLOQUE CONVOLUCIONAL REUTILIZABLE
# ─────────────────────────────────────────────────────────────────────
def convolutional_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int] = (3, 3),
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    block_id: int = 1
) -> tf.Tensor:
    """
    Bloque convolucional modular:
        Conv2D → BatchNormalization → ReLU → MaxPooling2D → Dropout

    La BatchNormalization estabiliza el entrenamiento al normalizar
    las activaciones de cada capa, permitiendo learning rates más altos
    y reduciendo la sensibilidad a la inicialización de pesos.

    El Dropout en capas convolucionales fuerza a la red a aprender
    representaciones robustas que no dependan de un único rango
    de frecuencias para su decisión de clasificación.

    Args:
        x:           Tensor de entrada.
        filters:     Número de filtros convolucionales.
        kernel_size: Tamaño del kernel (ancho × alto).
        dropout_rate: Tasa de Dropout (0.0 a 1.0).
        l2_reg:      Factor de regularización L2 en los pesos convolucionales.
        block_id:    Identificador para nombrado de capas.

    Returns:
        Tensor procesado por el bloque.
    """
    prefix = f"block{block_id}"

    # ── Convolución 1 ──
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",                          # Mantiene dimensiones espaciales
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,                          # BatchNorm absorbe el bias
        name=f"{prefix}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu1")(x)

    # ── Convolución 2 (doble conv para riqueza representacional) ──
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,
        name=f"{prefix}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu2")(x)

    # ── Reducción espacial ──
    x = layers.MaxPooling2D(
        pool_size=POOL_SIZE,
        strides=POOL_SIZE,
        name=f"{prefix}_pool"
    )(x)

    # ── Regularización ──
    x = layers.Dropout(rate=dropout_rate, name=f"{prefix}_dropout")(x)

    return x


# ─────────────────────────────────────────────────────────────────────
# ARQUITECTURA PRINCIPAL: XSIVE CNN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────
def build_xsive_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    conv_filters: list = None,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: list = None,
    learning_rate: float = LEARNING_RATE,
    l2_reg: float = 1e-4
) -> Model:
    """
    Construye y compila el clasificador binario CNN de Xsive.

    Arquitectura completa:

    ┌─────────────────────────────────────────────────────────┐
    │  Input: (128, 128, 1) — Mel Spectrogram monocanal       │
    ├─────────────────────────────────────────────────────────┤
    │  Block 1: Conv2D(32)  × 2 → BN → ReLU → Pool(2×2) → DO │
    │  Block 2: Conv2D(64)  × 2 → BN → ReLU → Pool(2×2) → DO │
    │  Block 3: Conv2D(128) × 2 → BN → ReLU → Pool(2×2) → DO │
    │  Block 4: Conv2D(256) × 2 → BN → ReLU → GlobalAvgPool  │
    ├─────────────────────────────────────────────────────────┤
    │  Dense(256) → BN → ReLU → Dropout(0.5)                  │
    │  Dense(64)  → BN → ReLU → Dropout(0.3)                  │
    ├─────────────────────────────────────────────────────────┤
    │  Output: Dense(1, Sigmoid) → P(Clase 1) ∈ [0.0, 1.0]   │
    └─────────────────────────────────────────────────────────┘

    Justificación de elecciones arquitectónicas:
      · Dobles conv por bloque: captura patrones a múltiples escalas
      · BatchNorm: convergencia más rápida, menor dependencia del LR
      · GlobalAveragePooling: reduce parámetros vs Flatten, más robusto
      · L2 regularization: penaliza pesos grandes, evita memorización
      · Binary Crossentropy: función de pérdida correcta para salida Sigmoid
      · Adam (lr=1e-4): converge suavemente en paisajes de pérdida no convexos

    Args:
        input_shape:   Shape del tensor de entrada (H, W, C).
        conv_filters:  Lista de filtros por bloque convolucional.
        dropout_rate:  Tasa de Dropout en bloques convolucionales.
        dense_units:   Lista de neuronas en capas densas.
        learning_rate: Learning rate inicial de Adam.
        l2_reg:        Factor de regularización L2.

    Returns:
        Modelo Keras compilado, listo para entrenamiento.
    """
    if conv_filters is None:
        conv_filters = CONV_FILTERS + [256]   # [32, 64, 128, 256]
    if dense_units is None:
        dense_units  = DENSE_UNITS            # [256, 64]

    # ── Capa de Entrada ──
    inputs = keras.Input(shape=input_shape, name="mel_spectrogram_input")

    # ── Data Augmentation inline (solo activa en training) ──
    # Pequeñas perturbaciones para simular variaciones del espectrograma
    x = layers.RandomFlip(
        mode="horizontal",
        name="aug_flip"
    )(inputs)
    x = layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05,
        name="aug_translate"
    )(x)

    # ── Bloque Convolucional 1: Detectores de bajo nivel ──
    # Aprende: franjas de energía, onset de frecuencias, contornos armónicos
    x = convolutional_block(
        x, filters=conv_filters[0],
        dropout_rate=dropout_rate * 0.75,   # 0.3 — menos regularización al inicio
        l2_reg=l2_reg,
        block_id=1
    )

    # ── Bloque Convolucional 2: Detectores de nivel medio ──
    # Aprende: densidades frecuenciales, texturas de síntesis, patrones de brillo
    x = convolutional_block(
        x, filters=conv_filters[1],
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        block_id=2
    )

    # ── Bloque Convolucional 3: Detectores de alto nivel ──
    # Aprende: firmas espectrales globales, distribución de energía >8kHz
    x = convolutional_block(
        x, filters=conv_filters[2],
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        block_id=3
    )

    # ── Bloque Convolucional 4: Abstracción máxima ──
    # Con 256 filtros detecta las configuraciones espectrales más complejas
    # Sin pooling final — GlobalAveragePooling colapsa las dimensiones
    x = layers.Conv2D(
        filters=conv_filters[3],
        kernel_size=KERNEL_SIZE,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,
        name="block4_conv1"
    )(x)
    x = layers.BatchNormalization(name="block4_bn1")(x)
    x = layers.Activation("relu", name="block4_relu1")(x)

    x = layers.Conv2D(
        filters=conv_filters[3],
        kernel_size=KERNEL_SIZE,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,
        name="block4_conv2"
    )(x)
    x = layers.BatchNormalization(name="block4_bn2")(x)
    x = layers.Activation("relu", name="block4_relu2")(x)

    # ── Global Average Pooling ──
    # Reemplaza Flatten+Dense grande: promedia cada mapa de características
    # Reduce de (N, H', W', 256) a (N, 256) con mucho menos parámetros
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # ── Capas Densas (Clasificador) ──
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units=units,
            kernel_regularizer=regularizers.l2(l2_reg),
            use_bias=False,
            name=f"dense_{units}"
        )(x)
        x = layers.BatchNormalization(name=f"dense_bn_{i}")(x)
        x = layers.Activation("relu", name=f"dense_relu_{i}")(x)
        # Mayor dropout en capa densa más grande
        drop = 0.5 if i == 0 else 0.3
        x = layers.Dropout(rate=drop, name=f"dense_dropout_{i}")(x)

    # ── Capa de Salida: Clasificación Binaria ──
    # Sigmoid: mapea cualquier valor real a probabilidad en [0.0, 1.0]
    # Una sola neurona → P(track pertenece a Clase 1 / Aprobado)
    outputs = layers.Dense(
        units=1,
        activation="sigmoid",
        dtype="float32",              # Forzar float32 para estabilidad numérica
        name="classification_output"
    )(x)

    # ── Construcción del Modelo ──
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="XsiveCNNClassifier"
    )

    # ── Compilación ──
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,              # Momentum del gradiente (default)
            beta_2=0.999,            # Momentum del cuadrado del gradiente
            epsilon=1e-7,            # Previene división por cero
            clipnorm=1.0             # Gradient clipping para estabilidad
        ),
        loss=keras.losses.BinaryCrossentropy(
            from_logits=False,       # La salida YA es probabilidad (Sigmoid aplicado)
            label_smoothing=0.05     # Suavizado ligero para evitar sobreconfianza
        ),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            keras.metrics.AUC(name="auc", curve="ROC"),
            keras.metrics.Precision(name="precision", thresholds=0.85),
            keras.metrics.Recall(name="recall", thresholds=0.85),
        ]
    )

    return model


# ─────────────────────────────────────────────────────────────────────
# UTILIDADES DE MODELO
# ─────────────────────────────────────────────────────────────────────
def get_model_summary(model: Model) -> str:
    """Retorna el resumen del modelo como string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def count_parameters(model: Model) -> dict:
    """Cuenta parámetros totales, entrenables y no entrenables."""
    total       = model.count_params()
    trainable   = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    non_train   = sum(
        tf.size(w).numpy() for w in model.non_trainable_weights
    )
    return {
        "total":         total,
        "trainable":     trainable,
        "non_trainable": non_train
    }


def save_model(model: Model, filepath: Optional[str] = None):
    """Guarda el modelo completo en formato Keras nativo."""
    if filepath is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        filepath = os.path.join(MODELS_DIR, MODEL_FILENAME)
    model.save(filepath)
    logger.info(f"✓ Modelo guardado en: {filepath}")


def load_model(filepath: Optional[str] = None) -> Model:
    """Carga un modelo Keras desde disco."""
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, MODEL_FILENAME)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    model = keras.models.load_model(filepath)
    logger.info(f"✓ Modelo cargado desde: {filepath}")
    return model


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN STANDALONE: Inspeccionar la arquitectura
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  XSIVE CNN — Inspección de Arquitectura")
    print("═" * 65 + "\n")

    model = build_xsive_cnn()
    model.summary()

    params = count_parameters(model)
    print(f"\n  Parámetros totales     : {params['total']:,}")
    print(f"  Parámetros entrenables : {params['trainable']:,}")
    print(f"  Parámetros fijos (BN)  : {params['non_trainable']:,}")
    print(f"\n  Input shape  : {INPUT_SHAPE}")
    print(f"  Output shape : (batch, 1) — Probabilidad Sigmoid")
    print(f"\n  Umbral de clasificación: >= 0.85 → APROBADO")
