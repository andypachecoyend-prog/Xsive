"""
╔══════════════════════════════════════════════════════════════════════╗
║             XSIVE AUDIO CLASSIFIER — CONFIGURATION MODULE           ║
║      Motor de Curaduría Espectral para la Ecología de Sonido Xsive  ║
╚══════════════════════════════════════════════════════════════════════╝

Centraliza todos los hiperparámetros y rutas del pipeline, garantizando
reproducibilidad total entre sesiones de entrenamiento.
"""

import os

# ─────────────────────────────────────────────────────────────────────
# RUTAS DEL SISTEMA DE ARCHIVOS
# ─────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "..", "data")          # Raíz de los audios crudos
CLASS1_DIR        = os.path.join(DATA_DIR, "vanguardista")        # Clase 1: Aprobado
CLASS0_DIR        = os.path.join(DATA_DIR, "antiguo")             # Clase 0: Denegado
PROCESSED_DIR     = os.path.join(BASE_DIR, "..", "processed")     # Tensores preprocesados (.npy)
MODELS_DIR        = os.path.join(BASE_DIR, "..", "models")        # Modelos entrenados (.h5 / .keras)
LOGS_DIR          = os.path.join(BASE_DIR, "..", "logs")          # TensorBoard / CSV logs
REPORTS_DIR       = os.path.join(BASE_DIR, "..", "reports")       # Matrices de confusión / reportes

MODEL_FILENAME    = "xsive_cnn_classifier.keras"
WEIGHTS_FILENAME  = "xsive_best_weights.weights.h5"
SCALER_FILENAME   = "mel_scaler.pkl"                              # StandardScaler serializado

# ─────────────────────────────────────────────────────────────────────
# PARÁMETROS DE AUDIO Y SEGMENTACIÓN
# ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 22050          # Hz — estándar para análisis espectral
SEGMENT_DURATION  = 30             # Segundos — ventana central de análisis
OFFSET_STRATEGY   = "center"       # Extraer desde el centro de la pista
MONO              = True           # Forzar canal único para la CNN

# Número de muestras por segmento
N_SAMPLES         = SAMPLE_RATE * SEGMENT_DURATION   # = 661500

# ─────────────────────────────────────────────────────────────────────
# PARÁMETROS DEL ESPECTROGRAMA DE MEL
# ─────────────────────────────────────────────────────────────────────
N_FFT             = 2048           # Tamaño de la ventana FFT
HOP_LENGTH        = 512            # Desplazamiento entre ventanas consecutivas
N_MELS            = 128            # Bins de frecuencia Mel — resolución vertical
F_MAX             = 11000          # Hz máximo — captura "brillo digital" de Clase 1
F_MIN             = 20             # Hz mínimo — excluye ruido de DC offset
WINDOW_FUNCTION   = "hann"         # Ventana de análisis para reducir spectral leakage

# Dimensiones resultantes del espectrograma Mel:
# time_steps = ceil(N_SAMPLES / HOP_LENGTH) = ~1292
# frequency_bins = N_MELS = 128
# Shape tensor: (1292, 128, 1)

# ─────────────────────────────────────────────────────────────────────
# PARÁMETROS DE MFCCs
# ─────────────────────────────────────────────────────────────────────
N_MFCC            = 20             # Coeficientes cepstrales (13–20 recomendado)
# Shape MFCCs: (20, 1292) → se usará como canal adicional o análisis paralelo

# ─────────────────────────────────────────────────────────────────────
# PARÁMETROS DE ENERGÍA RMS
# ─────────────────────────────────────────────────────────────────────
RMS_FRAME_LENGTH  = 2048           # Longitud de ventana para cálculo RMS
RMS_HOP_LENGTH    = 512            # Desplazamiento para cálculo RMS

# ─────────────────────────────────────────────────────────────────────
# ARQUITECTURA DE LA CNN
# ─────────────────────────────────────────────────────────────────────
INPUT_SHAPE       = (128, 128, 1)  # (time_steps_resized, frequency_bins, channels)
# Se redimensiona a 128x128 para uniformidad de input

CONV_FILTERS      = [32, 64, 128]  # Filtros por bloque convolucional
KERNEL_SIZE       = (3, 3)         # Kernel estándar para patrones locales
POOL_SIZE         = (2, 2)         # Reducción 2x en cada dimensión espacial
DROPOUT_RATE      = 0.4            # Tasa de regularización por Dropout
DENSE_UNITS       = [256, 64]      # Neuronas en capas densas intermedias

# ─────────────────────────────────────────────────────────────────────
# PARÁMETROS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 100            # EarlyStopping interrumpirá antes si converge
LEARNING_RATE     = 1e-4           # Adam LR — conservador para patrones sutiles
VALIDATION_SPLIT  = 0.15           # 15% del training set para validación interna
TEST_SPLIT        = 0.15           # 15% del dataset total para evaluación final
RANDOM_SEED       = 42             # Reproducibilidad garantizada

# ─────────────────────────────────────────────────────────────────────
# CALLBACKS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE   = 15         # Épocas sin mejora antes de detener
EARLY_STOP_MONITOR    = "val_loss"
LR_REDUCE_FACTOR      = 0.5        # Reducir LR a la mitad al estancarse
LR_REDUCE_PATIENCE    = 7          # Épocas antes de reducir LR
LR_REDUCE_MIN         = 1e-7       # LR mínimo permitido

# ─────────────────────────────────────────────────────────────────────
# UMBRAL DE CLASIFICACIÓN (INFERENCE)
# ─────────────────────────────────────────────────────────────────────
APPROVAL_THRESHOLD    = 0.85       # Sigmoid >= 0.85 → Clase 1 (Aprobado)
# Por debajo del umbral → Clase 0 (Denegado)

# ─────────────────────────────────────────────────────────────────────
# ETIQUETAS DE CLASE
# ─────────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: "DENEGADO (Antiguo/Minimalista)",
    1: "APROBADO (Vanguardista)"
}

# ─────────────────────────────────────────────────────────────────────
# UTILIDAD: Crear directorios si no existen
# ─────────────────────────────────────────────────────────────────────
def ensure_directories():
    """Crea todos los directorios del proyecto si no existen."""
    dirs = [DATA_DIR, CLASS1_DIR, CLASS0_DIR, PROCESSED_DIR,
            MODELS_DIR, LOGS_DIR, REPORTS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[CONFIG] ✓ Estructura de directorios verificada.")
