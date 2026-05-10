"""
╔══════════════════════════════════════════════════════════════════════╗
║      XSIVE AUDIO CLASSIFIER — FASE 4: TRAINING PIPELINE            ║
║          Entrenamiento, Callbacks y Visualización de Curvas         ║
╚══════════════════════════════════════════════════════════════════════╝

Orquesta el ciclo de entrenamiento completo con:
  · EarlyStopping          — Para cuando val_loss deja de mejorar
  · ModelCheckpoint        — Guarda el mejor modelo durante training
  · ReduceLROnPlateau      — Reduce learning rate dinámicamente
  · TensorBoard Logger     — Curvas de pérdida / accuracy en tiempo real
  · Curvas de aprendizaje  — Matplotlib PNG guardado en /reports
"""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Backend no-interactivo para generar PNGs en servidor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras

from config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    EARLY_STOP_PATIENCE, EARLY_STOP_MONITOR,
    LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE, LR_REDUCE_MIN,
    MODELS_DIR, LOGS_DIR, REPORTS_DIR,
    MODEL_FILENAME, WEIGHTS_FILENAME,
    RANDOM_SEED
)
from model import build_xsive_cnn, save_model, count_parameters

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("XsiveTrainer")
tf.random.set_seed(RANDOM_SEED)

# Timestamp para identificar la sesión de entrenamiento
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────
# CALLBACKS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────
def build_callbacks(weights_path: str, tensorboard_dir: str) -> list:
    """
    Construye la lista de callbacks para el ciclo de entrenamiento.

    Callbacks configurados:
      1. EarlyStopping       : Detiene el entrenamiento si val_loss
                               no mejora durante EARLY_STOP_PATIENCE épocas.
                               Restaura los mejores pesos automáticamente.

      2. ModelCheckpoint     : Guarda los pesos cada vez que val_loss
                               alcanza un nuevo mínimo histórico.

      3. ReduceLROnPlateau   : Reduce el learning rate en un factor
                               LR_REDUCE_FACTOR si val_loss se estanca.
                               Actúa como "enfriamiento gradual" del optimizer.

      4. TensorBoard         : Registra métricas, histogramas de pesos
                               y grafos del modelo para visualización.

      5. CSVLogger           : Guarda el historial de entrenamiento en
                               un CSV para análisis post-training.

    Args:
        weights_path:     Ruta donde guardar los mejores pesos (.h5).
        tensorboard_dir:  Directorio para los logs de TensorBoard.

    Returns:
        Lista de callbacks de Keras.
    """
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    csv_log_path = os.path.join(REPORTS_DIR, f"training_log_{SESSION_ID}.csv")

    callbacks = [
        # ── EarlyStopping ──
        keras.callbacks.EarlyStopping(
            monitor=EARLY_STOP_MONITOR,
            patience=EARLY_STOP_PATIENCE,
            min_delta=1e-4,              # Mejora mínima aceptable
            restore_best_weights=True,   # Restaura pesos del mejor epoch
            mode="min",
            verbose=1
        ),

        # ── ModelCheckpoint ──
        keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor=EARLY_STOP_MONITOR,
            save_best_only=True,         # Solo guarda si hay mejora
            save_weights_only=True,      # Solo pesos, no arquitectura
            mode="min",
            verbose=1
        ),

        # ── ReduceLROnPlateau ──
        keras.callbacks.ReduceLROnPlateau(
            monitor=EARLY_STOP_MONITOR,
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=LR_REDUCE_MIN,
            min_delta=1e-5,
            mode="min",
            verbose=1
        ),

        # ── TensorBoard ──
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,            # Histogramas de pesos cada época
            write_graph=True,
            update_freq="epoch"
        ),

        # ── CSVLogger ──
        keras.callbacks.CSVLogger(
            filename=csv_log_path,
            separator=",",
            append=False
        ),
    ]

    logger.info(f"  Callbacks configurados (session: {SESSION_ID})")
    logger.info(f"  TensorBoard: tensorboard --logdir {tensorboard_dir}")
    logger.info(f"  CSV log: {csv_log_path}")

    return callbacks


# ─────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────
def train_xsive_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    model:   Optional[keras.Model] = None,
    epochs:  int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    class_weight: Optional[Dict] = None
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Ejecuta el ciclo de entrenamiento completo del clasificador Xsive.

    Flujo de entrenamiento:
      1. Construir el modelo CNN (si no se provee uno)
      2. Configurar callbacks
      3. Calcular class weights si hay desbalance
      4. Ejecutar model.fit()
      5. Guardar modelo final
      6. Generar curvas de aprendizaje

    Args:
        X_train:      Tensores de entrenamiento (N, 128, 128, 1).
        y_train:      Etiquetas de entrenamiento (N,).
        X_val:        Tensores de validación (M, 128, 128, 1).
        y_val:        Etiquetas de validación (M,).
        model:        Modelo Keras precompilado (opcional).
        epochs:       Número máximo de épocas.
        batch_size:   Tamaño del batch de entrenamiento.
        class_weight: Dict de pesos de clase para desbalance {0: w0, 1: w1}.

    Returns:
        Tuple (modelo_entrenado, history_objeto)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # ── Rutas ──
    weights_path    = os.path.join(MODELS_DIR, WEIGHTS_FILENAME)
    tensorboard_dir = os.path.join(LOGS_DIR, f"session_{SESSION_ID}")

    # ── Construir modelo si no se provee ──
    if model is None:
        logger.info("\n[TRAINING] Construyendo arquitectura CNN...")
        model = build_xsive_cnn()

    # ── Resumen del modelo ──
    params = count_parameters(model)
    logger.info(
        f"\n[TRAINING] ═══ Resumen de Configuración ═══\n"
        f"  Parámetros totales : {params['total']:,}\n"
        f"  Training samples   : {X_train.shape[0]}\n"
        f"  Validation samples : {X_val.shape[0]}\n"
        f"  Input shape        : {X_train.shape[1:]}\n"
        f"  Epochs máximos     : {epochs}\n"
        f"  Batch size         : {batch_size}\n"
        f"  Learning rate      : {LEARNING_RATE}\n"
        f"  EarlyStopping pat  : {EARLY_STOP_PATIENCE}\n"
        f"  Session ID         : {SESSION_ID}"
    )

    # ── Calcular class weights automáticamente si hay desbalance ──
    if class_weight is None:
        n_total  = len(y_train)
        n_class0 = np.sum(y_train == 0)
        n_class1 = np.sum(y_train == 1)

        if abs(n_class0 - n_class1) / n_total > 0.05:
            # Desbalance > 5% → aplicar pesos inversamente proporcionales
            weight_for_0 = (1 / n_class0) * (n_total / 2.0)
            weight_for_1 = (1 / n_class1) * (n_total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}
            logger.info(
                f"  Class weights aplicados: "
                f"{{0: {weight_for_0:.3f}, 1: {weight_for_1:.3f}}}"
            )
        else:
            logger.info("  Dataset balanceado — sin class weights")

    # ── Callbacks ──
    callbacks = build_callbacks(weights_path, tensorboard_dir)

    # ── ENTRENAMIENTO ──
    logger.info("\n[TRAINING] ⚡ Iniciando entrenamiento...\n")
    start_time = datetime.now()

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
        verbose=1
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    best_epoch = np.argmin(history.history["val_loss"]) + 1

    logger.info(
        f"\n[TRAINING] ✓ Entrenamiento completado:\n"
        f"  Épocas ejecutadas  : {len(history.history['loss'])}\n"
        f"  Mejor época        : #{best_epoch}\n"
        f"  Mejor val_loss     : {min(history.history['val_loss']):.4f}\n"
        f"  Mejor val_accuracy : {max(history.history['val_accuracy']):.4f}\n"
        f"  Mejor val_auc      : {max(history.history['val_auc']):.4f}\n"
        f"  Tiempo total       : {elapsed:.1f}s ({elapsed/60:.1f} min)"
    )

    # ── Guardar modelo final ──
    final_model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    model.save(final_model_path)
    logger.info(f"\n  ✓ Modelo final guardado: {final_model_path}")

    # ── Guardar historial como JSON ──
    history_path = os.path.join(REPORTS_DIR, f"history_{SESSION_ID}.json")
    history_serializable = {k: [float(v) for v in vals]
                            for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)

    # ── Generar curvas de aprendizaje ──
    plot_training_curves(history, session_id=SESSION_ID)

    return model, history


# ─────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN DE CURVAS DE APRENDIZAJE
# ─────────────────────────────────────────────────────────────────────
def plot_training_curves(
    history: keras.callbacks.History,
    session_id: str = "",
    save_dir: str = REPORTS_DIR
):
    """
    Genera y guarda un gráfico de 4 paneles con las curvas de entrenamiento.

    Paneles:
      1. Binary Cross-Entropy Loss (train vs val)
      2. Accuracy (train vs val)
      3. AUC-ROC (train vs val)
      4. Learning Rate a lo largo de las épocas

    Args:
        history:    Objeto History retornado por model.fit().
        session_id: Identificador de la sesión para el nombre del archivo.
        save_dir:   Directorio donde guardar el PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    # Paleta de colores del proyecto
    COLOR_TRAIN = "#7C3AED"       # Violeta — entrenamiento
    COLOR_VAL   = "#10B981"       # Verde esmeralda — validación
    COLOR_BEST  = "#F59E0B"       # Ámbar — mejor época

    best_epoch = np.argmin(hist["val_loss"])

    fig = plt.figure(figsize=(16, 10), facecolor="#0F0F1A")
    fig.suptitle(
        "XSIVE CNN — Curvas de Aprendizaje",
        fontsize=16, fontweight="bold", color="white",
        fontfamily="monospace", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    axes_config = [
        ("loss",     "val_loss",     "Binary Cross-Entropy Loss",  gs[0, 0]),
        ("accuracy", "val_accuracy", "Accuracy",                   gs[0, 1]),
        ("auc",      "val_auc",      "AUC-ROC",                    gs[1, 0]),
    ]

    for train_key, val_key, title, grid_pos in axes_config:
        ax = fig.add_subplot(grid_pos)
        ax.set_facecolor("#1A1A2E")

        if train_key in hist:
            ax.plot(epochs, hist[train_key], color=COLOR_TRAIN,
                    linewidth=2, label="Training", alpha=0.9)
        if val_key in hist:
            ax.plot(epochs, hist[val_key], color=COLOR_VAL,
                    linewidth=2, label="Validation", linestyle="--", alpha=0.9)

        # Marcar mejor época
        if val_key in hist:
            ax.axvline(
                x=best_epoch + 1,
                color=COLOR_BEST, linewidth=1.5,
                linestyle=":", alpha=0.8,
                label=f"Best epoch #{best_epoch + 1}"
            )

        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Época", color="#94A3B8", fontsize=9)
        ax.tick_params(colors="#94A3B8")
        ax.spines[:].set_color("#334155")
        ax.legend(facecolor="#1E293B", edgecolor="#334155",
                  labelcolor="white", fontsize=8)
        ax.grid(alpha=0.15, color="#334155")

    # Panel 4: Learning Rate
    if "lr" in hist:
        ax_lr = fig.add_subplot(gs[1, 1])
        ax_lr.set_facecolor("#1A1A2E")
        ax_lr.plot(epochs, hist["lr"], color="#EC4899",
                   linewidth=2, alpha=0.9)
        ax_lr.set_title("Learning Rate", color="white", fontsize=11, pad=8)
        ax_lr.set_xlabel("Época", color="#94A3B8", fontsize=9)
        ax_lr.set_yscale("log")
        ax_lr.tick_params(colors="#94A3B8")
        ax_lr.spines[:].set_color("#334155")
        ax_lr.grid(alpha=0.15, color="#334155")

    filename = f"training_curves_{session_id}.png" if session_id else "training_curves.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0F0F1A")
    plt.close()

    logger.info(f"  ✓ Curvas de aprendizaje guardadas: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN STANDALONE (para reentrenamiento con splits pregenerados)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import load_splits_from_disk

    print("\n" + "═" * 65)
    print("  XSIVE — Entrenamiento con splits pregenerados")
    print("═" * 65)

    print("\n  Cargando splits desde disco...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits_from_disk()

    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape}")

    model, history = train_xsive_model(X_train, y_train, X_val, y_val)

    print(f"\n  Entrenamiento completado.")
    print(f"  Mejor val_loss: {min(history.history['val_loss']):.4f}")
