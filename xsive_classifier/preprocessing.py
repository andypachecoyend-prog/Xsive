"""
╔══════════════════════════════════════════════════════════════════════╗
║      XSIVE AUDIO CLASSIFIER — FASE 2: PREPROCESSING & TENSORS      ║
║               Normalización, Reshaping y Serialización              ║
╚══════════════════════════════════════════════════════════════════════╝

Convierte las matrices de características crudas en tensores 3D
normalizados compatibles con la arquitectura CNN de Keras.

Pipeline de preprocesamiento:
  1. Resize del Mel Spectrogram → target_size (128×128)
  2. Z-score normalization con StandardScaler (fit en train, transform en test)
  3. Reshape → (batch, height, width, 1) para Conv2D
  4. Persistencia de tensores en disco (.npy) para reentrenamiento rápido
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2  # Para resize de espectrogramas (más eficiente que librosa)

from config import (
    INPUT_SHAPE, PROCESSED_DIR, MODELS_DIR,
    SCALER_FILENAME, TEST_SPLIT, VALIDATION_SPLIT, RANDOM_SEED
)

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("XsivePreprocessor")

# Dimensión objetivo del tensor de entrada a la CNN
TARGET_H = INPUT_SHAPE[0]   # 128 (filas: frequency bins)
TARGET_W = INPUT_SHAPE[1]   # 128 (columnas: time steps)


# ─────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL: PREPROCESADOR DE TENSORES
# ─────────────────────────────────────────────────────────────────────
class XsivePreprocessor:
    """
    Transforma las características espectrales crudas en tensores 3D
    normalizados listos para la CNN.

    La CNN trata el Mel Spectrogram como una imagen en escala de grises:
      · Eje Y (altura)    → Frecuencias Mel (128 bins)
      · Eje X (anchura)   → Tiempo en frames
      · Canal (profundidad) → Único canal (como imagen B/N)

    El StandardScaler se ajusta ÚNICAMENTE en el training set para
    evitar fuga de datos (data leakage) hacia test/validation.
    """

    def __init__(self):
        self.scaler        = StandardScaler()
        self.scaler_fitted = False
        self.scaler_path   = os.path.join(MODELS_DIR, SCALER_FILENAME)

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 1: RESIZE DEL ESPECTROGRAMA
    # ────────────────────────────────────────────────────────────────
    def resize_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Redimensiona el Mel Spectrogram a la resolución objetivo (128×128).

        La CNN requiere que todas las imágenes tengan dimensiones idénticas.
        OpenCV ofrece interpolación bilineal de alta calidad para este fin.

        Args:
            mel_spec: Matriz (n_mels, time_steps) en dB.

        Returns:
            Matriz resizeada (TARGET_H, TARGET_W) con float32.
        """
        # mel_spec shape: (128, T) — necesitamos (TARGET_H, TARGET_W)
        resized = cv2.resize(
            mel_spec.astype(np.float32),
            (TARGET_W, TARGET_H),           # OpenCV: (width, height)
            interpolation=cv2.INTER_LINEAR  # Interpolación bilineal
        )
        return resized  # Shape: (128, 128)

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 2: RESHAPE A TENSOR 3D
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def add_channel_dimension(spectrogram_2d: np.ndarray) -> np.ndarray:
        """
        Añade la dimensión de canal para compatibilidad con Conv2D.

        Keras Conv2D espera tensores de forma (height, width, channels).
        Para imágenes en escala de grises (monocanal), channels=1.

        Args:
            spectrogram_2d: Matriz (height, width).

        Returns:
            Tensor (height, width, 1).
        """
        return np.expand_dims(spectrogram_2d, axis=-1)

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 3: PREPARAR ARRAY DE TENSORES DESDE FEATURES LIST
    # ────────────────────────────────────────────────────────────────
    def features_to_tensors(
        self,
        features_list: List[Dict]
    ) -> np.ndarray:
        """
        Convierte la lista de dicts de características en un array numpy 4D.

        Proceso por cada track:
          1. Tomar mel_spectrogram (128, T)
          2. Resize → (128, 128)
          3. Añadir canal → (128, 128, 1)
          4. Apilar en batch → (N, 128, 128, 1)

        Args:
            features_list: Lista de dicts retornados por extract_all_features().

        Returns:
            Array numpy de forma (N, TARGET_H, TARGET_W, 1).
        """
        tensors = []
        for feat_dict in features_list:
            mel_spec = feat_dict["mel_spectrogram"]          # (128, T)
            resized  = self.resize_spectrogram(mel_spec)     # (128, 128)
            tensor   = self.add_channel_dimension(resized)   # (128, 128, 1)
            tensors.append(tensor)

        X = np.array(tensors, dtype=np.float32)
        logger.info(f"  Tensores generados: {X.shape} | dtype: {X.dtype}")
        return X  # (N, 128, 128, 1)

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 4: NORMALIZACIÓN Z-SCORE (STANDARD SCALER)
    # ────────────────────────────────────────────────────────────────
    def fit_and_normalize(self, X_train: np.ndarray) -> np.ndarray:
        """
        Ajusta el StandardScaler sobre el training set y normaliza.

        Z-score: z = (x - μ) / σ
        Resultado: distribución con media≈0 y desviación estándar≈1.

        IMPORTANTE: Solo se llama fit_transform() sobre el TRAINING SET.
        El test set y nuevas inferencias usan solo transform().

        Args:
            X_train: Tensor de entrenamiento (N, 128, 128, 1).

        Returns:
            Tensor normalizado con la misma forma.
        """
        N, H, W, C = X_train.shape

        # Aplanar a 2D para el scaler: (N, H*W*C)
        X_flat = X_train.reshape(N, -1)

        # Ajustar y transformar
        X_normalized_flat = self.scaler.fit_transform(X_flat)

        # Restaurar shape original
        X_normalized = X_normalized_flat.reshape(N, H, W, C)
        self.scaler_fitted = True

        logger.info(
            f"  StandardScaler ajustado | "
            f"μ_mean={self.scaler.mean_.mean():.4f} | "
            f"σ_mean={self.scaler.scale_.mean():.4f}"
        )
        return X_normalized.astype(np.float32)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica el StandardScaler ya ajustado a nuevos datos (val / test / inferencia).

        Args:
            X: Tensor a normalizar (N, 128, 128, 1).

        Returns:
            Tensor normalizado con la misma forma.

        Raises:
            RuntimeError: Si el scaler no ha sido ajustado previamente.
        """
        if not self.scaler_fitted:
            raise RuntimeError(
                "El StandardScaler no ha sido ajustado. "
                "Llama primero a fit_and_normalize() sobre el training set."
            )
        N, H, W, C = X.shape
        X_flat      = X.reshape(N, -1)
        X_norm_flat = self.scaler.transform(X_flat)
        return X_norm_flat.reshape(N, H, W, C).astype(np.float32)

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 5: PERSISTENCIA DEL SCALER
    # ────────────────────────────────────────────────────────────────
    def save_scaler(self):
        """Serializa el StandardScaler entrenado a disco (.pkl)."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"  ✓ Scaler guardado en: {self.scaler_path}")

    def load_scaler(self):
        """Carga el StandardScaler serializado desde disco."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler no encontrado en: {self.scaler_path}. "
                "Ejecuta primero el entrenamiento."
            )
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.scaler_fitted = True
        logger.info(f"  ✓ Scaler cargado desde: {self.scaler_path}")


# ─────────────────────────────────────────────────────────────────────
# FUNCIÓN DE ALTO NIVEL: PIPELINE COMPLETO DE PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────
def build_dataset_splits(
    features_list: List[Dict],
    labels_list: List[int],
    preprocessor: Optional[XsivePreprocessor] = None
) -> Tuple[np.ndarray, np.ndarray,   # X_train, y_train
           np.ndarray, np.ndarray,   # X_val,   y_val
           np.ndarray, np.ndarray,   # X_test,  y_test
           XsivePreprocessor]:
    """
    Construye los tres conjuntos (train / validation / test) normalizados.

    División del dataset:
      - Test set (15%):     evaluación final, nunca visto durante entrenamiento
      - Validation set (15% del remanente): monitoreo de EarlyStopping
      - Training set (70%): aprendizaje de la CNN

    La normalización se ajusta ÚNICAMENTE sobre el training set.

    Args:
        features_list: Lista de dicts de características (de feature_extraction.py).
        labels_list:   Lista de etiquetas enteras (0 o 1).
        preprocessor:  Instancia de XsivePreprocessor (opcional).

    Returns:
        Tuple con los 6 arrays numpy (X/y para train, val, test) +
        el preprocessor ajustado.
    """
    if preprocessor is None:
        preprocessor = XsivePreprocessor()

    labels = np.array(labels_list, dtype=np.int32)

    # ── Paso 1: Construir tensores crudos ──
    logger.info("\n[PREPROCESAMIENTO] Construyendo tensores...")
    X = preprocessor.features_to_tensors(features_list)
    y = labels

    # ── Paso 2: Split train/test estratificado ──
    logger.info("[PREPROCESAMIENTO] Dividiendo dataset...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y              # Garantiza proporción 50/50 en cada split
    )

    # ── Paso 3: Split train/validation ──
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train_val
    )

    logger.info(
        f"  Training  : {X_train.shape[0]} samples "
        f"({y_train.sum()} Clase1 | {(y_train==0).sum()} Clase0)\n"
        f"  Validation: {X_val.shape[0]} samples "
        f"({y_val.sum()} Clase1 | {(y_val==0).sum()} Clase0)\n"
        f"  Test      : {X_test.shape[0]} samples "
        f"({y_test.sum()} Clase1 | {(y_test==0).sum()} Clase0)"
    )

    # ── Paso 4: Normalizar (fit solo en training) ──
    logger.info("[PREPROCESAMIENTO] Normalizando con StandardScaler...")
    X_train = preprocessor.fit_and_normalize(X_train)
    X_val   = preprocessor.normalize(X_val)
    X_test  = preprocessor.normalize(X_test)

    # ── Paso 5: Guardar scaler para inferencia futura ──
    preprocessor.save_scaler()

    logger.info(
        f"[PREPROCESAMIENTO] ✓ Tensores finales:\n"
        f"  X_train : {X_train.shape} | y_train: {y_train.shape}\n"
        f"  X_val   : {X_val.shape}   | y_val  : {y_val.shape}\n"
        f"  X_test  : {X_test.shape}  | y_test : {y_test.shape}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor


# ─────────────────────────────────────────────────────────────────────
# PERSISTENCIA DE SPLITS EN DISCO (opcional, para reentrenamiento rápido)
# ─────────────────────────────────────────────────────────────────────
def save_splits_to_disk(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    output_dir: str = PROCESSED_DIR
):
    """Guarda los splits numpy en disco para evitar reextracción."""
    os.makedirs(output_dir, exist_ok=True)
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    for name, array in splits.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, array)
    logger.info(f"✓ Splits guardados en: {output_dir}")


def load_splits_from_disk(
    input_dir: str = PROCESSED_DIR
) -> Tuple[np.ndarray, ...]:
    """Carga los splits numpy desde disco."""
    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    arrays = []
    for name in names:
        path = os.path.join(input_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Split '{name}.npy' no encontrado en {input_dir}. "
                "Ejecuta primero el pipeline completo."
            )
        arrays.append(np.load(path))
    logger.info(f"✓ Splits cargados desde: {input_dir}")
    return tuple(arrays)
