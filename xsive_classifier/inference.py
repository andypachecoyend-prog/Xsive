"""
╔══════════════════════════════════════════════════════════════════════╗
║      XSIVE AUDIO CLASSIFIER — FASE 5: INFERENCE & VALIDATION        ║
║       Motor de Predicción + Matriz de Confusión + Reportes          ║
╚══════════════════════════════════════════════════════════════════════╝

Módulo de inferencia y evaluación del clasificador entrenado:

  · predict_track()      → Clasifica un archivo de audio individual
  · evaluate_test_set()  → Evalúa el test set completo con métricas
  · plot_confusion_matrix() → Visualización de TP/FP/TN/FN
  · generate_full_report()  → Informe completo en texto + PNG
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import tensorflow as tf
from tensorflow import keras

from config import (
    APPROVAL_THRESHOLD, CLASS_LABELS,
    MODELS_DIR, REPORTS_DIR,
    MODEL_FILENAME, SCALER_FILENAME
)
from feature_extraction import XsiveFeatureExtractor
from preprocessing import XsivePreprocessor

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("XsiveInference")


# ─────────────────────────────────────────────────────────────────────
# CLASE: MOTOR DE INFERENCIA
# ─────────────────────────────────────────────────────────────────────
class XsivePredictor:
    """
    Motor de inferencia end-to-end para clasificar tracks de audio individuales.

    Carga el modelo entrenado y el scaler, aplica el mismo pipeline
    de preprocesamiento que en el entrenamiento, y retorna la
    clasificación con probabilidad y veredicto legible.

    Uso:
        predictor = XsivePredictor()
        result = predictor.predict_track("/path/to/track.wav")
        print(result)
    """

    def __init__(
        self,
        model_path:  Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        self.model_path  = model_path  or os.path.join(MODELS_DIR, MODEL_FILENAME)
        self.scaler_path = scaler_path or os.path.join(MODELS_DIR, SCALER_FILENAME)

        self.model       = None
        self.extractor   = XsiveFeatureExtractor()
        self.preprocessor = XsivePreprocessor()

        self._load_artifacts()

    def _load_artifacts(self):
        """Carga el modelo y el scaler entrenados desde disco."""
        # Cargar modelo
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modelo no encontrado en: {self.model_path}\n"
                "Ejecuta main.py primero para entrenar el clasificador."
            )
        self.model = keras.models.load_model(self.model_path)
        logger.info(f"✓ Modelo cargado: {self.model_path}")

        # Cargar scaler
        self.preprocessor.load_scaler()
        logger.info(f"✓ Scaler cargado: {self.scaler_path}")

    # ────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL: CLASIFICAR UN TRACK INDIVIDUAL
    # ────────────────────────────────────────────────────────────────
    def predict_track(
        self,
        filepath: str,
        verbose: bool = True
    ) -> Dict:
        """
        Clasifica un archivo de audio y retorna el veredicto de Xsive.

        Pipeline de inferencia:
          1. Cargar audio y extraer segmento central (30s)
          2. Generar Mel Spectrogram en dB
          3. Resize → (128, 128)
          4. Normalizar con el StandardScaler entrenado
          5. Añadir dimensiones batch y canal: (1, 128, 128, 1)
          6. Pasar por la CNN → valor Sigmoid ∈ [0.0, 1.0]
          7. Aplicar umbral: >= 0.85 → APROBADO, < 0.85 → DENEGADO

        Args:
            filepath: Ruta al archivo de audio (.wav, .mp3, .flac, etc.)
            verbose:  Imprimir resultado en consola.

        Returns:
            Dict con:
              - filename      : Nombre del archivo
              - probability   : Valor Sigmoid crudo [0.0, 1.0]
              - predicted_class: 0 o 1
              - verdict       : "APROBADO" | "DENEGADO"
              - label         : Descripción completa de la clase
              - confidence    : "ALTA" | "MEDIA" | "BAJA"
        """
        filename = Path(filepath).name

        # ── 1. Extracción de características ──
        features = self.extractor.extract_all_features(filepath)
        if features is None:
            logger.error(f"✗ No se pudo procesar: {filename}")
            return {"error": f"No se pudo procesar el archivo: {filename}"}

        # ── 2. Construir tensor ──
        mel_spec = features["mel_spectrogram"]
        resized  = self.preprocessor.resize_spectrogram(mel_spec)
        tensor   = self.preprocessor.add_channel_dimension(resized)

        # ── 3. Normalizar ──
        X = np.array([tensor], dtype=np.float32)   # (1, 128, 128, 1)
        X_norm = self.preprocessor.normalize(X)

        # ── 4. Inferencia ──
        probability = float(self.model.predict(X_norm, verbose=0)[0][0])

        # ── 5. Aplicar umbral de aprobación ──
        predicted_class = 1 if probability >= APPROVAL_THRESHOLD else 0
        verdict = "APROBADO ✓" if predicted_class == 1 else "DENEGADO ✗"
        label   = CLASS_LABELS[predicted_class]

        # ── 6. Nivel de confianza ──
        if probability >= 0.90 or probability <= 0.10:
            confidence = "ALTA"
        elif probability >= 0.75 or probability <= 0.25:
            confidence = "MEDIA"
        else:
            confidence = "BAJA"   # Zona gris (próximo al umbral 0.5)

        result = {
            "filename":       filename,
            "probability":    probability,
            "predicted_class": predicted_class,
            "verdict":        verdict,
            "label":          label,
            "confidence":     confidence,
            "threshold_used": APPROVAL_THRESHOLD
        }

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  Track   : {filename}")
            print(f"  P(Clase1): {probability:.4f}  ({probability*100:.1f}%)")
            print(f"  Umbral  : >= {APPROVAL_THRESHOLD} → Aprobado")
            print(f"  Veredicto: {verdict}")
            print(f"  Clase    : {label}")
            print(f"  Confianza: {confidence}")
            print(f"{'─'*55}")

        return result

    # ────────────────────────────────────────────────────────────────
    # MÉTODO: EVALUAR SET DE PRUEBA COMPLETO
    # ────────────────────────────────────────────────────────────────
    def evaluate_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evalúa el clasificador sobre el test set con métricas completas.

        Métricas calculadas:
          · Accuracy general
          · Precision, Recall, F1-Score por clase
          · Matriz de Confusión (TP, TN, FP, FN)
          · AUC-ROC
          · AUC-PR (Precision-Recall)
          · Threshold analysis

        Args:
            X_test: Tensores del test set ya normalizados (N, 128, 128, 1).
            y_test: Etiquetas reales del test set (N,).

        Returns:
            Dict con todas las métricas calculadas.
        """
        logger.info("\n[EVALUATION] Ejecutando inferencia sobre test set...")

        # ── Predicciones ──
        y_prob = self.model.predict(X_test, batch_size=32, verbose=1).flatten()
        y_pred = (y_prob >= APPROVAL_THRESHOLD).astype(int)

        # ── Confusion Matrix ──
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # ── Classification Report ──
        report = classification_report(
            y_test, y_pred,
            target_names=["Clase 0 (Antiguo)", "Clase 1 (Vanguardista)"],
            digits=4
        )

        # ── AUC-ROC ──
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # ── AUC-PR ──
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)

        metrics = {
            "total_samples":   len(y_test),
            "true_positives":  int(tp),
            "true_negatives":  int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "accuracy":        float((tp + tn) / (tp + tn + fp + fn)),
            "precision_c1":    float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            "recall_c1":       float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            "f1_c1":           float(2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 0,
            "specificity":     float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            "roc_auc":         float(roc_auc),
            "avg_precision":   float(avg_precision),
            "classification_report": report,
            "confusion_matrix":      cm,
            "y_prob":                y_prob,
            "y_pred":                y_pred,
            "y_test":                y_test,
            "fpr":                   fpr,
            "tpr":                   tpr,
            "precision_vals":        precision_vals,
            "recall_vals":           recall_vals
        }

        # ── Imprimir resumen ──
        print("\n" + "═"*65)
        print("  XSIVE — REPORTE DE EVALUACIÓN DEL TEST SET")
        print("═"*65)
        print(f"\n  Muestras totales : {metrics['total_samples']}")
        print(f"  Verdaderos Positivos (TP): {tp:4d}  → Vanguardistas correctamente aprobados")
        print(f"  Verdaderos Negativos (TN): {tn:4d}  → Antiguos correctamente denegados")
        print(f"  Falsos Positivos   (FP): {fp:4d}  → Antiguos aprobados por error")
        print(f"  Falsos Negativos   (FN): {fn:4d}  → Vanguardistas denegados por error")
        print(f"\n  Accuracy    : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Precision   : {metrics['precision_c1']:.4f}")
        print(f"  Recall      : {metrics['recall_c1']:.4f}")
        print(f"  F1-Score    : {metrics['f1_c1']:.4f}")
        print(f"  Specificity : {metrics['specificity']:.4f}")
        print(f"  AUC-ROC     : {metrics['roc_auc']:.4f}")
        print(f"  Avg. Prec.  : {metrics['avg_precision']:.4f}")
        print(f"\n  Umbral de clasificación: {APPROVAL_THRESHOLD}")
        print("\n" + report)

        return metrics


# ─────────────────────────────────────────────────────────────────────
# VISUALIZACIONES DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm: np.ndarray,
    session_id: str = "",
    save_dir: str = REPORTS_DIR
):
    """
    Genera una visualización premium de la Matriz de Confusión.

    La matriz muestra los 4 cuadrantes clave:
      TP (arriba izq): Vanguardistas correctamente identificados
      TN (abajo der):  Antiguos correctamente rechazados
      FP (arriba der): Falsos positivos (ruido del sistema)
      FN (abajo izq):  Falsos negativos (pérdida de curaduría)

    Args:
        cm:          Matriz de confusión 2×2 de sklearn.
        session_id:  Identificador para el nombre del archivo.
        save_dir:    Directorio de guardado.
    """
    os.makedirs(save_dir, exist_ok=True)

    tn, fp, fn, tp = cm.ravel()
    labels = ["Clase 0\n(Antiguo)", "Clase 1\n(Vanguardista)"]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    # Heatmap con seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Purples",
        linewidths=2,
        linecolor="#1A1A2E",
        annot_kws={"size": 20, "weight": "bold", "color": "white"},
        ax=ax
    )

    # Anotaciones descriptivas en cada celda
    cell_labels = [
        (0, 0, "TN\n(Rechazados correctos)",  "#10B981"),
        (0, 1, "FP\n(Falsos Positivos)",       "#EF4444"),
        (1, 0, "FN\n(Falsos Negativos)",        "#F59E0B"),
        (1, 1, "TP\n(Aprobados correctos)",     "#7C3AED"),
    ]

    for row, col, label_text, color in cell_labels:
        ax.text(
            col + 0.5, row + 0.75, label_text,
            ha="center", va="center",
            fontsize=8, color=color, alpha=0.9,
            fontweight="bold"
        )

    ax.set_title(
        "XSIVE CNN — Matriz de Confusión\nTest Set Evaluation",
        color="white", fontsize=13, fontweight="bold",
        fontfamily="monospace", pad=15
    )
    ax.set_xlabel("Predicción del Modelo", color="#94A3B8", fontsize=11, labelpad=10)
    ax.set_ylabel("Etiqueta Real",         color="#94A3B8", fontsize=11, labelpad=10)
    ax.tick_params(colors="#94A3B8", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)

    filename = f"confusion_matrix_{session_id}.png" if session_id else "confusion_matrix.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()

    logger.info(f"✓ Matriz de confusión guardada: {save_path}")
    return save_path


def plot_roc_and_pr_curves(
    metrics: Dict,
    session_id: str = "",
    save_dir: str = REPORTS_DIR
):
    """
    Genera las curvas ROC y Precision-Recall en un panel combinado.

    Args:
        metrics:    Dict retornado por evaluate_test_set().
        session_id: Identificador de sesión.
        save_dir:   Directorio de guardado.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0F0F1A")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1A1A2E")
        ax.spines[:].set_color("#334155")
        ax.tick_params(colors="#94A3B8")
        ax.grid(alpha=0.15, color="#334155")

    # ── Curva ROC ──
    ax1.plot(
        metrics["fpr"], metrics["tpr"],
        color="#7C3AED", linewidth=2.5,
        label=f"AUC-ROC = {metrics['roc_auc']:.4f}"
    )
    ax1.plot([0, 1], [0, 1], color="#475569", linewidth=1, linestyle="--",
             label="Random Classifier")
    ax1.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.15, color="#7C3AED")
    ax1.set_title("Curva ROC", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Tasa de Falsos Positivos (FPR)", color="#94A3B8")
    ax1.set_ylabel("Tasa de Verdaderos Positivos (TPR)", color="#94A3B8")
    ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])

    # ── Curva Precision-Recall ──
    ax2.plot(
        metrics["recall_vals"], metrics["precision_vals"],
        color="#10B981", linewidth=2.5,
        label=f"Avg Precision = {metrics['avg_precision']:.4f}"
    )
    ax2.fill_between(metrics["recall_vals"], metrics["precision_vals"],
                     alpha=0.15, color="#10B981")
    ax2.axhline(y=0.5, color="#475569", linewidth=1, linestyle="--",
                label="Baseline (0.5)")
    ax2.set_title("Curva Precision-Recall", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Recall", color="#94A3B8")
    ax2.set_ylabel("Precision", color="#94A3B8")
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])

    fig.suptitle(
        "XSIVE CNN — Análisis de Rendimiento del Clasificador",
        color="white", fontsize=13, fontweight="bold",
        fontfamily="monospace"
    )

    filename = f"roc_pr_curves_{session_id}.png" if session_id else "roc_pr_curves.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()

    logger.info(f"✓ Curvas ROC/PR guardadas: {save_path}")
    return save_path


def plot_probability_distribution(
    metrics: Dict,
    session_id: str = "",
    save_dir: str = REPORTS_DIR
):
    """
    Visualiza la distribución de probabilidades predichas por clase.

    Permite detectar:
      · Separabilidad de las dos clases
      · Zona de incertidumbre alrededor del umbral
      · Calibración del modelo

    Args:
        metrics:    Dict de evaluate_test_set() con y_prob e y_test.
        session_id: Identificador de sesión.
        save_dir:   Directorio de guardado.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_prob = metrics["y_prob"]
    y_test = metrics["y_test"]

    prob_class0 = y_prob[y_test == 0]
    prob_class1 = y_prob[y_test == 1]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0F0F1A")
    ax.set_facecolor("#1A1A2E")

    bins = np.linspace(0, 1, 50)

    ax.hist(prob_class0, bins=bins, alpha=0.7, color="#EF4444",
            label=f"Clase 0 — Antiguo/Denegado (n={len(prob_class0)})",
            edgecolor="#0F0F1A", linewidth=0.5)
    ax.hist(prob_class1, bins=bins, alpha=0.7, color="#7C3AED",
            label=f"Clase 1 — Vanguardista/Aprobado (n={len(prob_class1)})",
            edgecolor="#0F0F1A", linewidth=0.5)

    # Línea de umbral
    ax.axvline(
        x=APPROVAL_THRESHOLD,
        color="#F59E0B", linewidth=2.5,
        linestyle="--",
        label=f"Umbral de aprobación: {APPROVAL_THRESHOLD}"
    )

    ax.set_title(
        "Distribución de Probabilidades por Clase",
        color="white", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("P(Clase 1 / Aprobado) — Salida Sigmoid", color="#94A3B8", fontsize=11)
    ax.set_ylabel("Frecuencia", color="#94A3B8", fontsize=11)
    ax.tick_params(colors="#94A3B8")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white", fontsize=9)
    ax.grid(alpha=0.15, color="#334155")

    filename = f"prob_distribution_{session_id}.png" if session_id else "prob_distribution.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
    plt.close()

    logger.info(f"✓ Distribución de probabilidades guardada: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────
# FUNCIÓN DE ALTO NIVEL: REPORTE COMPLETO
# ─────────────────────────────────────────────────────────────────────
def generate_full_report(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictor: Optional[XsivePredictor] = None,
    session_id: str = ""
) -> Dict:
    """
    Genera el reporte de evaluación completo del clasificador Xsive.

    Produce:
      1. Métricas calculadas (accuracy, precision, recall, F1, AUC)
      2. Matriz de Confusión (PNG)
      3. Curvas ROC y Precision-Recall (PNG)
      4. Distribución de probabilidades (PNG)
      5. Reporte de texto guardado en /reports

    Args:
        X_test:     Tensores del test set normalizados.
        y_test:     Etiquetas reales del test set.
        predictor:  Instancia de XsivePredictor (opcional).
        session_id: Identificador de la sesión de entrenamiento.

    Returns:
        Dict con todas las métricas y rutas a los archivos generados.
    """
    if predictor is None:
        predictor = XsivePredictor()

    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Evaluar sobre test set ──
    metrics = predictor.evaluate_test_set(X_test, y_test)

    # ── Generar visualizaciones ──
    logger.info("\n[REPORT] Generando visualizaciones...")

    cm_path   = plot_confusion_matrix(metrics["confusion_matrix"],
                                      session_id, REPORTS_DIR)
    roc_path  = plot_roc_and_pr_curves(metrics, session_id, REPORTS_DIR)
    prob_path = plot_probability_distribution(metrics, session_id, REPORTS_DIR)

    # ── Guardar reporte de texto ──
    report_path = os.path.join(REPORTS_DIR, f"evaluation_report_{session_id}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("XSIVE CNN CLASSIFIER — EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Session ID   : {session_id}\n")
        f.write(f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Samples : {metrics['total_samples']}\n")
        f.write(f"Threshold    : {APPROVAL_THRESHOLD}\n\n")
        f.write("── CONFUSION MATRIX ──\n")
        f.write(f"  TP: {metrics['true_positives']:4d}  |  FP: {metrics['false_positives']:4d}\n")
        f.write(f"  FN: {metrics['false_negatives']:4d}  |  TN: {metrics['true_negatives']:4d}\n\n")
        f.write("── METRICS ──\n")
        f.write(f"  Accuracy    : {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision   : {metrics['precision_c1']:.4f}\n")
        f.write(f"  Recall      : {metrics['recall_c1']:.4f}\n")
        f.write(f"  F1-Score    : {metrics['f1_c1']:.4f}\n")
        f.write(f"  Specificity : {metrics['specificity']:.4f}\n")
        f.write(f"  AUC-ROC     : {metrics['roc_auc']:.4f}\n")
        f.write(f"  Avg Prec.   : {metrics['avg_precision']:.4f}\n\n")
        f.write("── CLASSIFICATION REPORT ──\n")
        f.write(metrics["classification_report"])
        f.write(f"\n── GENERATED FILES ──\n")
        f.write(f"  Confusion Matrix    : {cm_path}\n")
        f.write(f"  ROC/PR Curves       : {roc_path}\n")
        f.write(f"  Prob. Distribution  : {prob_path}\n")

    logger.info(f"\n✓ Reporte guardado: {report_path}")
    print(f"\n  Reporte completo guardado en: {report_path}")

    metrics["report_paths"] = {
        "evaluation_report":  report_path,
        "confusion_matrix":   cm_path,
        "roc_pr_curves":      roc_path,
        "prob_distribution":  prob_path
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN STANDALONE: Clasificar un track de audio desde CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from preprocessing import load_splits_from_disk

    if len(sys.argv) > 1:
        # Modo inferencia individual
        audio_path = sys.argv[1]
        print(f"\n{'═'*55}")
        print(f"  XSIVE — Clasificador de Audio")
        print(f"{'═'*55}")
        predictor = XsivePredictor()
        result = predictor.predict_track(audio_path, verbose=True)
    else:
        # Modo evaluación del test set
        print("\n  Cargando test set desde disco...")
        splits = load_splits_from_disk()
        X_test, y_test = splits[4], splits[5]

        predictor = XsivePredictor()
        metrics = generate_full_report(X_test, y_test, predictor)

        print(f"\n  AUC-ROC : {metrics['roc_auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
