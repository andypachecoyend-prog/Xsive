"""
╔══════════════════════════════════════════════════════════════════════╗
║         XSIVE AUDIO CLASSIFIER — MAIN PIPELINE ORCHESTRATOR         ║
║          Pipeline Completo: Extracción → Entrenamiento → Reporte    ║
╚══════════════════════════════════════════════════════════════════════╝

Punto de entrada principal del sistema. Ejecuta el pipeline completo
de extremo a extremo:

  1. Verificar estructura de directorios y librerías
  2. FASE 1: Extraer características espectrales del dataset completo
  3. FASE 2: Preprocesar y construir splits train/val/test normalizados
  4. FASE 3: Construir arquitectura CNN
  5. FASE 4: Entrenar el clasificador con callbacks
  6. FASE 5: Evaluar en test set y generar reportes

Uso:
  python main.py                              # Proceso completo
  python main.py --from-processed             # Saltar extracción (usar .npy en disco)
  python main.py --predict /path/to/audio.wav # Solo inferencia
  python main.py --eval-only                  # Solo evaluar modelo existente

Salida:
  models/xsive_cnn_classifier.keras   → Modelo entrenado
  models/xsive_best_weights.h5        → Mejores pesos
  models/mel_scaler.pkl               → StandardScaler serializado
  reports/                            → PNGs + CSV + TXT de métricas
  logs/                               → TensorBoard logs
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# ASEGURAR QUE EL DIRECTORIO DEL MÓDULO ESTÁ EN sys.path
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    ensure_directories,
    CLASS1_DIR, CLASS0_DIR,
    PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
    APPROVAL_THRESHOLD, RANDOM_SEED
)
from feature_extraction import XsiveFeatureExtractor, extract_dataset_features
from preprocessing import (
    XsivePreprocessor,
    build_dataset_splits,
    save_splits_to_disk,
    load_splits_from_disk
)
from model import build_xsive_cnn
from train import train_xsive_model
from inference import (
    XsivePredictor,
    generate_full_report
)

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING GLOBAL
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(
                os.path.dirname(SCRIPT_DIR), "logs",
                f"xsive_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            mode="w",
            encoding="utf-8",
            delay=True   # Crear archivo solo cuando se necesite
        )
    ]
)
logger = logging.getLogger("XsiveMain")


# ─────────────────────────────────────────────────────────────────────
# CABECERA DEL SISTEMA
# ─────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██╗  ██╗███████╗██╗██╗   ██╗███████╗                              ║
║   ╚██╗██╔╝██╔════╝██║██║   ██║██╔════╝                              ║
║    ╚███╔╝ ███████╗██║██║   ██║█████╗                                ║
║    ██╔██╗ ╚════██║██║╚██╗ ██╔╝██╔══╝                                ║
║   ██╔╝ ██╗███████║██║ ╚████╔╝ ███████╗                              ║
║   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝  ╚══════╝                              ║
║                                                                      ║
║   CNN Audio Binary Classifier — Spectral Curation Engine            ║
║   Vanguardista vs. Antiguo | Aprobado vs. Denegado                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES DE VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────
def check_dependencies():
    """Verifica que todas las librerías requeridas estén instaladas."""
    required = {
        "numpy":          "numpy",
        "librosa":        "librosa",
        "tensorflow":     "tensorflow",
        "sklearn":        "scikit-learn",
        "cv2":            "opencv-python",
        "matplotlib":     "matplotlib",
        "seaborn":        "seaborn",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\n❌ Librerías faltantes: {', '.join(missing)}")
        print(f"   Instalar con: pip install {' '.join(missing)}")
        sys.exit(1)

    import tensorflow as tf
    print(f"  TensorFlow version : {tf.__version__}")
    print(f"  GPU disponible     : {bool(tf.config.list_physical_devices('GPU'))}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU Memory Growth   : Habilitado ({len(gpus)} GPU(s))")


def check_dataset_structure():
    """Verifica que los directorios de datos existen y contienen archivos."""
    SUPPORTED = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}

    for cls_dir, cls_name in [(CLASS1_DIR, "Clase 1 (Vanguardista)"),
                               (CLASS0_DIR, "Clase 0 (Antiguo)")]:
        if not os.path.exists(cls_dir):
            print(f"\n⚠  Directorio no encontrado: {cls_dir}")
            print(f"   Crea el directorio y añade los tracks de {cls_name}")
            print(f"   Continuando de todas formas (esperando --from-processed)...")
            return False

        files = [
            f for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED
        ]
        print(f"  {cls_name}: {len(files)} tracks encontrados en {cls_dir}")

    return True


# ─────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────
def run_full_pipeline(from_processed: bool = False) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento de extremo a extremo.

    Args:
        from_processed: Si True, salta la extracción y carga splits desde disco.

    Returns:
        Dict con el modelo entrenado, métricas y rutas de artefactos.
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    logger.info(f"\n{'━'*65}")
    logger.info(f"  XSIVE — Iniciando Pipeline Completo | Session: {session_id}")
    logger.info(f"{'━'*65}\n")

    # ─────────────────────────────────────────
    # FASE 1 + 2: Extracción y Preprocesamiento
    # ─────────────────────────────────────────
    if from_processed:
        logger.info("⚡ Cargando splits pregenerados desde disco...")
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = load_splits_from_disk()
            preprocessor = XsivePreprocessor()
            preprocessor.load_scaler()
            logger.info("✓ Splits y scaler cargados exitosamente.\n")
        except FileNotFoundError as e:
            logger.error(f"✗ {e}")
            logger.error("  Ejecuta sin --from-processed para generar los splits.")
            sys.exit(1)
    else:
        # ── FASE 1: Extracción de características ──
        logger.info("━━━ FASE 1: Extracción Espectral ━━━")
        extractor = XsiveFeatureExtractor()
        features_list, labels_list = extract_dataset_features(
            class1_dir=CLASS1_DIR,
            class0_dir=CLASS0_DIR,
            extractor=extractor
        )

        if len(features_list) == 0:
            logger.error(
                "✗ No se encontraron archivos de audio. "
                "Verifica las rutas en config.py:\n"
                f"  CLASS1_DIR: {CLASS1_DIR}\n"
                f"  CLASS0_DIR: {CLASS0_DIR}"
            )
            sys.exit(1)

        # ── FASE 2: Preprocesamiento y splits ──
        logger.info("\n━━━ FASE 2: Preprocesamiento y Tensores ━━━")
        (X_train, y_train,
         X_val,   y_val,
         X_test,  y_test,
         preprocessor) = build_dataset_splits(features_list, labels_list)

        # Persistir splits en disco para reentrenamiento rápido
        save_splits_to_disk(X_train, y_train, X_val, y_val, X_test, y_test)
        logger.info("✓ Splits guardados en disco.\n")

    # ─────────────────────────────────────────
    # FASE 3: Arquitectura CNN
    # ─────────────────────────────────────────
    logger.info("━━━ FASE 3: Construcción de la CNN ━━━")
    model = build_xsive_cnn()
    logger.info(f"✓ Modelo construido: {model.count_params():,} parámetros\n")

    # ─────────────────────────────────────────
    # FASE 4: Entrenamiento
    # ─────────────────────────────────────────
    logger.info("━━━ FASE 4: Entrenamiento ━━━")
    model, history = train_xsive_model(
        X_train, y_train,
        X_val,   y_val,
        model=model
    )

    # ─────────────────────────────────────────
    # FASE 5: Evaluación y Reporte
    # ─────────────────────────────────────────
    logger.info("\n━━━ FASE 5: Evaluación y Reportes ━━━")
    predictor = XsivePredictor()
    metrics = generate_full_report(
        X_test, y_test,
        predictor=predictor,
        session_id=session_id
    )

    elapsed = time.time() - start_time

    logger.info(f"\n{'═'*65}")
    logger.info("  XSIVE CNN — Pipeline Completado Exitosamente")
    logger.info(f"{'═'*65}")
    logger.info(f"  Session ID     : {session_id}")
    logger.info(f"  Tiempo total   : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Accuracy (test): {metrics['accuracy']*100:.1f}%")
    logger.info(f"  AUC-ROC (test) : {metrics['roc_auc']:.4f}")
    logger.info(f"  F1-Score (C1)  : {metrics['f1_c1']:.4f}")
    logger.info(f"\n  Modelo guardado   : models/xsive_cnn_classifier.keras")
    logger.info(f"  Reportes PNG      : reports/")
    logger.info(f"  TensorBoard logs  : logs/session_{session_id}/")
    logger.info(f"\n  Para visualizar TensorBoard:")
    logger.info(f"  tensorboard --logdir logs/session_{session_id}")

    return {
        "model":      model,
        "history":    history,
        "metrics":    metrics,
        "session_id": session_id
    }


def run_single_inference(audio_path: str):
    """Clasifica un único archivo de audio y muestra el veredicto."""
    print(f"\n{'═'*60}")
    print(f"  XSIVE — Clasificador de Track Individual")
    print(f"{'═'*60}")

    if not os.path.exists(audio_path):
        print(f"\n❌ Archivo no encontrado: {audio_path}")
        sys.exit(1)

    predictor = XsivePredictor()
    result = predictor.predict_track(audio_path, verbose=True)

    print(f"\n  Clasificación completada.")
    return result


def run_eval_only():
    """Evalúa el modelo existente sobre el test set guardado."""
    print(f"\n{'═'*60}")
    print(f"  XSIVE — Evaluación sobre Test Set")
    print(f"{'═'*60}\n")

    splits = load_splits_from_disk()
    X_test, y_test = splits[4], splits[5]

    predictor = XsivePredictor()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = generate_full_report(X_test, y_test, predictor, session_id)

    print(f"\n  Evaluación completada.")
    print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  AUC-ROC:  {metrics['roc_auc']:.4f}")


# ─────────────────────────────────────────────────────────────────────
# CLI: ARGPARSE
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="XSIVE — Clasificador Binario de Audio CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                              → Pipeline completo
  python main.py --from-processed             → Reentrenar sin reextracción
  python main.py --predict track.wav          → Clasificar un track
  python main.py --eval-only                  → Evaluar modelo existente
        """
    )

    parser.add_argument(
        "--from-processed",
        action="store_true",
        help="Cargar splits .npy desde disco (saltar extracción de características)"
    )
    parser.add_argument(
        "--predict",
        type=str,
        metavar="AUDIO_PATH",
        help="Ruta a un archivo de audio para clasificar (solo inferencia)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluar el modelo entrenado sobre el test set guardado"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Verificar dependencias e información del sistema"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(BANNER)

    args = parse_args()

    # Crear directorios necesarios
    ensure_directories()
    os.makedirs(os.path.join(os.path.dirname(SCRIPT_DIR), "logs"), exist_ok=True)

    # ── Solo verificar dependencias ──
    if args.check_deps:
        print("\n  Verificando dependencias...")
        check_dependencies()
        print("\n  ✓ Todas las dependencias están instaladas.")
        sys.exit(0)

    # ── Verificar dependencias siempre ──
    check_dependencies()

    # ── Modo: Inferencia individual ──
    if args.predict:
        run_single_inference(args.predict)

    # ── Modo: Solo evaluación ──
    elif args.eval_only:
        run_eval_only()

    # ── Modo: Pipeline completo ──
    else:
        if not args.from_processed:
            print("\n  Verificando dataset...")
            check_dataset_structure()

        results = run_full_pipeline(from_processed=args.from_processed)

        print("\n" + "═"*65)
        print("  Pipeline finalizado con éxito.")
        print(f"  Modelo: models/{os.path.basename(results['model'].name)}")
        print(f"  Revisa reports/ para las visualizaciones generadas.")
        print("═"*65 + "\n")
