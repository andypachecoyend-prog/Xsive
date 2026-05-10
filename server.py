"""
╔══════════════════════════════════════════════════════════════════════╗
║         XSIVE — FLASK API SERVER (Backend de la Web UI)            ║
║     Expone el pipeline ML como REST API para el frontend            ║
╚══════════════════════════════════════════════════════════════════════╝

Endpoints:
  GET  /api/status          → Estado del modelo (cargado / sin entrenar)
  POST /api/predict         → Clasificar un audio (multipart/form-data)
  GET  /api/metrics         → Métricas del último test set
  GET  /api/spectrogram     → Imagen del espectrograma del último audio
  POST /api/train           → Iniciar entrenamiento (async, background)
  GET  /api/train/progress  → SSE stream de progreso del entrenamiento

Uso:
  python server.py
  Luego abrir: http://localhost:5000
"""

import os
import sys
import json
import time
import uuid
import base64
import logging
import tempfile
import threading
from io import BytesIO
from datetime import datetime
from pathlib import Path

# Añadir el directorio del clasificador al path
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_DIR = os.path.join(SCRIPT_DIR, "xsive_classifier")
if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)

from flask import (
    Flask, request, jsonify, send_from_directory,
    Response, stream_with_context
)
from flask_cors import CORS

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

# Importar módulos del clasificador
try:
    from config import (
        APPROVAL_THRESHOLD, CLASS_LABELS, MODELS_DIR,
        MODEL_FILENAME, SCALER_FILENAME, REPORTS_DIR, LOGS_DIR,
        CLASS1_DIR, CLASS0_DIR, PROCESSED_DIR, ensure_directories
    )
    from feature_extraction import XsiveFeatureExtractor
    from preprocessing import XsivePreprocessor
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    CLASSIFIER_AVAILABLE = False
    print(f"⚠ Módulos del clasificador no disponibles: {e}")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE FLASK
# ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=SCRIPT_DIR, static_url_path="")
CORS(app)  # Permitir requests desde el frontend

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("XsiveServer")

# Estado global compartido
MODEL_STATE = {
    "loaded":          False,
    "model":           None,
    "predictor":       None,
    "last_prediction": None,
    "last_spectrogram":None,   # Base64 PNG
    "training_active": False,
    "training_log":    [],
    "metrics":         None,
    "model_path":      os.path.join(MODELS_DIR if CLASSIFIER_AVAILABLE else "models",
                                    MODEL_FILENAME if CLASSIFIER_AVAILABLE else "model.keras")
}


# ─────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN: Cargar modelo si existe
# ─────────────────────────────────────────────────────────────────────
def initialize_model():
    """Intenta cargar el modelo entrenado al iniciar el servidor."""
    if not CLASSIFIER_AVAILABLE:
        return

    try:
        ensure_directories()
    except Exception:
        pass

    model_path = MODEL_STATE["model_path"]
    if os.path.exists(model_path):
        try:
            from inference import XsivePredictor
            predictor = XsivePredictor()
            MODEL_STATE["predictor"] = predictor
            MODEL_STATE["loaded"]    = True
            logger.info(f"✓ Modelo cargado automáticamente: {model_path}")

            # Cargar métricas si existen
            _load_latest_metrics()
        except Exception as e:
            logger.warning(f"⚠ No se pudo cargar el modelo: {e}")
    else:
        logger.info(f"ℹ Modelo no encontrado en: {model_path}")
        logger.info("  Usa el endpoint POST /api/train o entrena con main.py")


def _load_latest_metrics():
    """Carga el reporte de evaluación más reciente."""
    reports_dir = REPORTS_DIR if CLASSIFIER_AVAILABLE else "reports"
    if not os.path.exists(reports_dir):
        return
    reports = sorted(
        [f for f in os.listdir(reports_dir) if f.startswith("evaluation_report")],
        reverse=True
    )
    if not reports:
        return
    try:
        with open(os.path.join(reports_dir, reports[0]), "r", encoding="utf-8") as f:
            content = f.read()
        MODEL_STATE["metrics"] = {"raw_text": content}
        logger.info(f"  ✓ Métricas cargadas: {reports[0]}")
    except Exception as e:
        logger.warning(f"  ⚠ Error cargando métricas: {e}")


# ─────────────────────────────────────────────────────────────────────
# UTILIDAD: Generar imágenes de espectrograma
# ─────────────────────────────────────────────────────────────────────
def generate_spectrogram_image(audio_path: str) -> str:
    """
    Genera un espectrograma visual del audio y lo retorna como Base64 PNG.

    Produce un panel de 3 gráficos:
      1. Mel Spectrogram (dB)
      2. MFCCs (envolvente tímbrica)
      3. RMS Energy (dinámica)

    Returns:
        String Base64 del PNG generado.
    """
    try:
        sr      = 22050
        duration = 30

        # Calcular offset central
        total_dur = librosa.get_duration(path=audio_path)
        offset = max(0.0, total_dur / 2.0 - 15.0) if total_dur > 30 else 0.0

        audio, _ = librosa.load(audio_path, sr=sr, mono=True,
                                offset=offset, duration=duration)

        # Generar características
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=11000, hop_length=512)
        mel_db   = librosa.power_to_db(mel_spec, ref=np.max)

        mfccs    = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20,
                                         hop_length=512, fmax=11000)

        rms      = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
        rms_db   = librosa.amplitude_to_db(rms, ref=np.max)

        times    = librosa.frames_to_time(
            np.arange(mel_db.shape[1]), sr=sr, hop_length=512)

        # ── Figura ──
        fig = plt.figure(figsize=(14, 9), facecolor="#0A0A15")
        gs  = gridspec.GridSpec(3, 1, hspace=0.45, left=0.08, right=0.97,
                                top=0.92, bottom=0.08)

        panel_style = dict(facecolor="#111128")
        tick_kw     = dict(colors="#7C8DB0", labelsize=8)
        label_kw    = dict(color="#7C8DB0", fontsize=9)

        # ── Panel 1: Mel Spectrogram ──
        ax1 = fig.add_subplot(gs[0])
        ax1.set(**panel_style)
        img1 = ax1.imshow(
            mel_db, aspect="auto", origin="lower",
            extent=[times[0], times[-1], 0, 128],
            cmap="magma", interpolation="bilinear",
            vmin=-80, vmax=0
        )
        cb1 = plt.colorbar(img1, ax=ax1, pad=0.01)
        cb1.set_label("dB", color="#7C8DB0", fontsize=8)
        cb1.ax.tick_params(colors="#7C8DB0", labelsize=7)
        ax1.set_title("Mel Spectrogram", color="#C4B5FD", fontsize=10,
                       fontweight="bold", pad=6)
        ax1.set_ylabel("Frecuencia (Mel bins)", **label_kw)
        ax1.tick_params(**tick_kw)
        ax1.spines[:].set_color("#1E1E3A")

        # ── Panel 2: MFCCs ──
        ax2 = fig.add_subplot(gs[1])
        ax2.set(**panel_style)
        img2 = ax2.imshow(
            mfccs, aspect="auto", origin="lower",
            extent=[times[0], times[-1], 0, 20],
            cmap="coolwarm", interpolation="bilinear"
        )
        cb2 = plt.colorbar(img2, ax=ax2, pad=0.01)
        cb2.set_label("Amp.", color="#7C8DB0", fontsize=8)
        cb2.ax.tick_params(colors="#7C8DB0", labelsize=7)
        ax2.set_title("MFCCs (Envolvente Tímbrica)", color="#86EFAC", fontsize=10,
                       fontweight="bold", pad=6)
        ax2.set_ylabel("Coeficiente", **label_kw)
        ax2.tick_params(**tick_kw)
        ax2.spines[:].set_color("#1E1E3A")

        # ── Panel 3: RMS Energy ──
        ax3 = fig.add_subplot(gs[2])
        ax3.set(**panel_style)
        rms_times = librosa.frames_to_time(
            np.arange(rms_db.shape[1]), sr=sr, hop_length=512)
        ax3.fill_between(rms_times, rms_db[0], rms_db[0].min(),
                          alpha=0.4, color="#F59E0B")
        ax3.plot(rms_times, rms_db[0], color="#FBBF24", linewidth=1.2)
        ax3.set_title("Energía RMS (Dinámica)", color="#FCD34D", fontsize=10,
                       fontweight="bold", pad=6)
        ax3.set_ylabel("dBFS", **label_kw)
        ax3.set_xlabel("Tiempo (s)", **label_kw)
        ax3.tick_params(**tick_kw)
        ax3.spines[:].set_color("#1E1E3A")
        ax3.grid(alpha=0.12, color="#1E1E3A")

        fig.suptitle(f"Análisis Espectral — {Path(audio_path).name}",
                     color="white", fontsize=11, fontweight="bold",
                     fontfamily="monospace")

        # ── Codificar a Base64 ──
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor="#0A0A15")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    except Exception as e:
        logger.error(f"Error generando espectrograma: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# ENDPOINT: Servir la web UI (index.html)
# ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "index.html")


# ─────────────────────────────────────────────────────────────────────
# ENDPOINT: Estado del sistema
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    """Retorna el estado actual del sistema y el modelo."""
    model_path = MODEL_STATE["model_path"]
    return jsonify({
        "model_loaded":      MODEL_STATE["loaded"],
        "model_path":        model_path,
        "model_exists":      os.path.exists(model_path),
        "training_active":   MODEL_STATE["training_active"],
        "has_metrics":       MODEL_STATE["metrics"] is not None,
        "classifier_ready":  CLASSIFIER_AVAILABLE,
        "approval_threshold": APPROVAL_THRESHOLD if CLASSIFIER_AVAILABLE else 0.85,
        "server_time":       datetime.now().isoformat()
    })


# ─────────────────────────────────────────────────────────────────────
# ENDPOINT: Clasificar un track de audio
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Recibe un archivo de audio, genera el espectrograma y clasifica.

    Multipart form-data:
      file: archivo de audio (.wav, .mp3, .flac, etc.)

    Response JSON:
      {
        "filename":      "track.wav",
        "probability":   0.921,
        "verdict":       "APROBADO",
        "label":         "Clase 1 (Vanguardista)",
        "confidence":    "ALTA",
        "spectrogram":   "data:image/png;base64,...",
        "processing_ms": 1234
      }
    """
    if "file" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo de audio."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    # Guardar en archivo temporal
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        start_ms = time.time()

        # ── Generar espectrograma (siempre disponible) ──
        spectrogram_b64 = generate_spectrogram_image(tmp_path)

        # ── Clasificar con el modelo (si está cargado) ──
        if MODEL_STATE["loaded"] and MODEL_STATE["predictor"]:
            result = MODEL_STATE["predictor"].predict_track(tmp_path, verbose=False)
            probability = result["probability"]
            verdict     = "APROBADO" if result["predicted_class"] == 1 else "DENEGADO"
            label       = result["label"]
            confidence  = result["confidence"]
        else:
            # ── Modo demo: simular clasificación espectral ──
            import random
            random.seed(hash(file.filename) % 1000)
            probability = round(random.uniform(0.15, 0.98), 4)
            threshold   = APPROVAL_THRESHOLD if CLASSIFIER_AVAILABLE else 0.85
            verdict     = "APROBADO" if probability >= threshold else "DENEGADO"
            label       = ("Clase 1 (Vanguardista)" if probability >= threshold
                           else "Clase 0 (Antiguo/Minimalista)")
            if probability >= 0.90 or probability <= 0.10:
                confidence = "ALTA"
            elif probability >= 0.75 or probability <= 0.25:
                confidence = "MEDIA"
            else:
                confidence = "BAJA"

        elapsed_ms = int((time.time() - start_ms) * 1000)

        response = {
            "filename":       file.filename,
            "probability":    probability,
            "verdict":        verdict,
            "label":          label,
            "confidence":     confidence,
            "spectrogram":    spectrogram_b64,
            "processing_ms":  elapsed_ms,
            "model_used":     MODEL_STATE["loaded"],
            "threshold":      APPROVAL_THRESHOLD if CLASSIFIER_AVAILABLE else 0.85
        }

        MODEL_STATE["last_prediction"]  = response
        MODEL_STATE["last_spectrogram"] = spectrogram_b64

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# ENDPOINT: Métricas del modelo
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/metrics")
def api_metrics():
    """Retorna métricas del último test set evaluado."""
    metrics_dir = REPORTS_DIR if CLASSIFIER_AVAILABLE else "reports"

    # Buscar el CSV de historial de entrenamiento más reciente
    history_data = None
    if os.path.exists(metrics_dir):
        csvs = sorted(
            [f for f in os.listdir(metrics_dir) if f.startswith("training_log")],
            reverse=True
        )
        if csvs:
            import csv
            csv_path = os.path.join(metrics_dir, csvs[0])
            rows = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: float(v) if v else 0 for k, v in row.items()})
            history_data = rows

    # Buscar imágenes generadas
    images = {}
    if os.path.exists(metrics_dir):
        for prefix, key in [("confusion_matrix", "confusion_matrix"),
                             ("roc_pr_curves",    "roc_pr"),
                             ("prob_distribution","prob_dist"),
                             ("training_curves",  "training_curves")]:
            files = sorted(
                [f for f in os.listdir(metrics_dir) if f.startswith(prefix)],
                reverse=True
            )
            if files:
                img_path = os.path.join(metrics_dir, files[0])
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                images[key] = f"data:image/png;base64,{b64}"

    return jsonify({
        "has_metrics":   MODEL_STATE["metrics"] is not None,
        "history":       history_data,
        "images":        images,
        "metrics_text":  MODEL_STATE["metrics"].get("raw_text", "") if MODEL_STATE["metrics"] else ""
    })


# ─────────────────────────────────────────────────────────────────────
# ENDPOINT: Iniciar entrenamiento (background thread)
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def api_train():
    """Inicia el pipeline de entrenamiento completo en background."""
    if MODEL_STATE["training_active"]:
        return jsonify({"error": "Ya hay un entrenamiento en curso."}), 409

    if not CLASSIFIER_AVAILABLE:
        return jsonify({"error": "Módulos del clasificador no disponibles."}), 503

    from_processed = request.json.get("from_processed", False) if request.is_json else False

    def training_thread():
        MODEL_STATE["training_active"] = True
        MODEL_STATE["training_log"]    = []

        def log(msg):
            timestamp = datetime.now().strftime("%H:%M:%S")
            entry = {"time": timestamp, "msg": msg}
            MODEL_STATE["training_log"].append(entry)
            logger.info(f"[TRAIN] {msg}")

        try:
            log("🚀 Iniciando pipeline de entrenamiento Xsive...")
            time.sleep(0.5)

            if not from_processed:
                log("📂 FASE 1: Extrayendo características espectrales...")
                from feature_extraction import XsiveFeatureExtractor, extract_dataset_features
                extractor = XsiveFeatureExtractor()
                features_list, labels_list = extract_dataset_features(
                    CLASS1_DIR, CLASS0_DIR, extractor)
                log(f"✓ {len(features_list)} tracks procesados.")

                log("🔧 FASE 2: Preprocesando tensores y normalizando...")
                from preprocessing import build_dataset_splits, save_splits_to_disk
                (X_train, y_train, X_val, y_val,
                 X_test, y_test, _) = build_dataset_splits(features_list, labels_list)
                save_splits_to_disk(X_train, y_train, X_val, y_val, X_test, y_test)
                log(f"✓ Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")
            else:
                log("💾 Cargando splits pregenerados desde disco...")
                from preprocessing import load_splits_from_disk
                splits = load_splits_from_disk()
                X_train, y_train, X_val, y_val, X_test, y_test = splits
                log(f"✓ Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

            log("🏗 FASE 3: Construyendo arquitectura CNN...")
            from model import build_xsive_cnn
            model = build_xsive_cnn()
            log(f"✓ {model.count_params():,} parámetros totales.")

            log("⚡ FASE 4: Entrenando el clasificador...")
            from train import train_xsive_model
            model, history = train_xsive_model(X_train, y_train, X_val, y_val, model=model)
            best_val_loss = min(history.history["val_loss"])
            best_val_acc  = max(history.history["val_accuracy"])
            log(f"✓ Entrenamiento completado — val_loss: {best_val_loss:.4f} | val_acc: {best_val_acc:.4f}")

            log("📊 FASE 5: Evaluando en test set y generando reportes...")
            from inference import XsivePredictor, generate_full_report
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictor  = XsivePredictor()
            metrics    = generate_full_report(X_test, y_test, predictor, session_id)

            MODEL_STATE["predictor"] = predictor
            MODEL_STATE["loaded"]    = True
            _load_latest_metrics()

            log(f"✅ Pipeline completo — Accuracy: {metrics['accuracy']*100:.1f}% | AUC-ROC: {metrics['roc_auc']:.4f}")

        except Exception as e:
            log(f"❌ Error durante entrenamiento: {str(e)}")
            logger.error("Error en training_thread:", exc_info=True)
        finally:
            MODEL_STATE["training_active"] = False

    thread = threading.Thread(target=training_thread, daemon=True)
    thread.start()

    return jsonify({"status": "started", "message": "Entrenamiento iniciado en background."})


# ─────────────────────────────────────────────────────────────────────
# ENDPOINT: SSE — Progreso del entrenamiento en tiempo real
# ─────────────────────────────────────────────────────────────────────
@app.route("/api/train/progress")
def api_train_progress():
    """Server-Sent Events para monitorear el progreso del entrenamiento."""
    def event_stream():
        last_index = 0
        while True:
            log = MODEL_STATE["training_log"]
            if len(log) > last_index:
                for entry in log[last_index:]:
                    data = json.dumps(entry)
                    yield f"data: {data}\n\n"
                last_index = len(log)

            if not MODEL_STATE["training_active"] and last_index >= len(log):
                yield f"data: {json.dumps({'time': datetime.now().strftime('%H:%M:%S'), 'msg': '__DONE__'})}\n\n"
                break
            time.sleep(0.5)

    return Response(
        stream_with_context(event_stream()),
        content_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*"
        }
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║   XSIVE — Servidor Web del Clasificador de Audio    ║
╚══════════════════════════════════════════════════════╝
""")
    ensure_directories() if CLASSIFIER_AVAILABLE else None
    initialize_model()

    print(f"\n  🌐 Interfaz web: http://localhost:5000")
    print(f"  📡 API REST:     http://localhost:5000/api/status")
    print(f"  Ctrl+C para detener el servidor\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
