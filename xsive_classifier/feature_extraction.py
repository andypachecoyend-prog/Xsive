"""
╔══════════════════════════════════════════════════════════════════════╗
║         XSIVE AUDIO CLASSIFIER — FASE 1: FEATURE EXTRACTION        ║
║                  Audio DSP & Spectral Engineering                   ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline de extracción de características acústicas que transforma
archivos de audio crudos en representaciones matriciales computables.

Características extraídas:
  · Mel Spectrogram  → Mapa energético frecuencia/tiempo en escala dB
  · MFCCs            → Envolvente espectral / textura tímbrica
  · RMS Energy       → Perfil dinámico de amplitud
"""

import os
import math
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import librosa
import librosa.display

from config import (
    SAMPLE_RATE, SEGMENT_DURATION, N_SAMPLES,
    N_FFT, HOP_LENGTH, N_MELS, F_MAX, F_MIN, WINDOW_FUNCTION,
    N_MFCC, RMS_FRAME_LENGTH, RMS_HOP_LENGTH
)

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LOGGING
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("XsiveFeatureExtractor")
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL: EXTRACTOR DE CARACTERÍSTICAS
# ─────────────────────────────────────────────────────────────────────
class XsiveFeatureExtractor:
    """
    Extrae representaciones espectrales de archivos de audio para el
    clasificador binario CNN de Xsive.

    Cada archivo pasa por el siguiente pipeline DSP:
      1. Carga y resampling → 22050 Hz mono
      2. Extracción del segmento central de 30s
      3. Generación del Espectrograma de Mel (dB)
      4. Extracción de MFCCs (N_MFCC coeficientes)
      5. Cálculo de energía RMS por frame

    Returns:
        Dict con las tres matrices numpy listas para normalización.
    """

    def __init__(self):
        self.sr           = SAMPLE_RATE
        self.duration     = SEGMENT_DURATION
        self.n_samples    = N_SAMPLES
        self.n_fft        = N_FFT
        self.hop_length   = HOP_LENGTH
        self.n_mels       = N_MELS
        self.f_max        = F_MAX
        self.f_min        = F_MIN
        self.window       = WINDOW_FUNCTION
        self.n_mfcc       = N_MFCC

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 1: CARGA Y SEGMENTACIÓN CENTRAL
    # ────────────────────────────────────────────────────────────────
    def load_audio_segment(
        self,
        filepath: str,
        offset_strategy: str = "center"
    ) -> Optional[np.ndarray]:
        """
        Carga un archivo de audio y extrae el segmento central.

        La extracción central evita:
          · Intros silentes o con fade-in
          · Outros con fade-out
          · Ruido de vinilo o leader tape al inicio

        Args:
            filepath: Ruta absoluta al archivo de audio.
            offset_strategy: "center" (recomendado) o "beginning".

        Returns:
            Array numpy con las muestras de audio (mono, 22050 Hz),
            o None si el archivo no pudo cargarse.
        """
        try:
            # Obtener duración total sin cargar el archivo completo
            total_duration = librosa.get_duration(path=filepath)

            if total_duration < self.duration:
                # Si el archivo es más corto que el segmento objetivo,
                # cargarlo completo con padding de ceros
                logger.warning(
                    f"⚠ '{Path(filepath).name}' dura {total_duration:.1f}s "
                    f"(< {self.duration}s). Se usará padding de ceros."
                )
                audio, _ = librosa.load(filepath, sr=self.sr, mono=True)
                # Pad con ceros hasta N_SAMPLES
                audio = librosa.util.fix_length(audio, size=self.n_samples)
                return audio

            # Calcular offset para extracción central
            if offset_strategy == "center":
                center = total_duration / 2.0
                offset = max(0.0, center - (self.duration / 2.0))
            else:
                offset = 0.0

            # Cargar solo el segmento necesario (eficiente en memoria)
            audio, _ = librosa.load(
                filepath,
                sr=self.sr,
                mono=True,
                offset=offset,
                duration=float(self.duration)
            )

            # Asegurar exactamente N_SAMPLES muestras
            audio = librosa.util.fix_length(audio, size=self.n_samples)

            logger.debug(
                f"✓ '{Path(filepath).name}' — {self.duration}s desde {offset:.1f}s "
                f"| Shape: {audio.shape}"
            )
            return audio

        except Exception as e:
            logger.error(f"✗ Error cargando '{filepath}': {e}")
            return None

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 2: ESPECTROGRAMA DE MEL (dB)
    # ────────────────────────────────────────────────────────────────
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Genera el Espectrograma de Mel en escala logarítmica (dB).

        El Mel Spectrogram mapea la energía de la señal en el espacio
        frecuencia-tiempo usando la escala Mel, que aproxima la percepción
        auditiva humana (resolución alta en bajas frecuencias).

        La conversión a dB (power_to_db) es crucial porque:
          · Comprime el rango dinámico extremo de amplitudes (~120dB)
          · Hace que la CNN trabaje en rangos numéricos manejables
          · Resalta diferencias sutiles en el contenido de alta frecuencia
            que distinguen la Clase 1 (Vanguardista) de la Clase 0

        Args:
            audio: Array numpy de muestras de audio normalizadas.

        Returns:
            Matriz (n_mels, time_steps) con valores en dB.
            Shape esperada: (128, ~1292)
        """
        # Potencia espectral en escala Mel
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            window=self.window,
            power=2.0           # Potencia cuadrática (energía), no amplitud
        )

        # Conversión a escala logarítmica en decibelios
        # ref=np.max normaliza el pico a 0dB (Top Reference)
        mel_db = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max,
            top_db=80.0         # Rango dinámico máximo representado
        )

        logger.debug(f"  Mel Spectrogram shape: {mel_db.shape} | "
                     f"dB range: [{mel_db.min():.1f}, {mel_db.max():.1f}]")
        return mel_db  # Shape: (128, time_steps)

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 3: MFCCs — ENVOLVENTE ESPECTRAL
    # ────────────────────────────────────────────────────────────────
    def extract_mfccs(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrae los Mel-Frequency Cepstral Coefficients (MFCCs).

        Los MFCCs representan la envolvente espectral de la señal,
        capturando el "timbre" o textura tonal independientemente de
        la energía absoluta. Son herramientas fundamentales para
        distinguir diferentes materiales acústicos y timbres sintéticos.

        · MFCC 1: Energía total del espectro (DC component)
        · MFCC 2-5: Forma general del espectro (brillo/oscuridad)
        · MFCC 6-13: Detalles de textura tímbrica
        · MFCC 14-20: Microdetalles espectrales (útiles para síntesis digital)

        Args:
            audio: Array numpy de muestras de audio.

        Returns:
            Matriz (n_mfcc, time_steps) con coeficientes cepstrales.
            Shape esperada: (20, ~1292)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.f_min,
            fmax=self.f_max,
            n_mels=self.n_mels
        )

        # Incluir delta MFCCs para capturar variación temporal del timbre
        mfcc_delta = librosa.feature.delta(mfccs)        # Primera derivada
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)  # Segunda derivada

        # Apilar: MFCCs base + deltas temporales
        mfccs_full = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
        # Shape resultante: (60, time_steps) = 20 * 3

        logger.debug(f"  MFCCs (+ deltas) shape: {mfccs_full.shape}")
        return mfccs_full

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 4: ENERGÍA RMS — PERFIL DINÁMICO
    # ────────────────────────────────────────────────────────────────
    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Calcula la energía RMS (Root Mean Square) por frame temporal.

        El RMS refleja la amplitud percibida de la señal en cada
        ventana de tiempo, permitiendo al modelo razonar sobre:
          · Rango dinámico (dB) de la pista
          · Contraste entre secciones suaves y climáticas
          · Presencia de compresión excesiva (pistas "brick-walled")

        Args:
            audio: Array numpy de muestras de audio.

        Returns:
            Array 1D con los valores RMS por frame.
            Shape esperada: (1, ~1292) → se usará como feature adicional
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=RMS_FRAME_LENGTH,
            hop_length=RMS_HOP_LENGTH,
            center=True
        )

        # Convertir RMS a dB para consistencia con el espectrograma
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        logger.debug(f"  RMS Energy shape: {rms_db.shape} | "
                     f"dB range: [{rms_db.min():.1f}, {rms_db.max():.1f}]")
        return rms_db  # Shape: (1, time_steps)

    # ────────────────────────────────────────────────────────────────
    # MÉTODO 5: PIPELINE COMPLETO (POR ARCHIVO)
    # ────────────────────────────────────────────────────────────────
    def extract_all_features(
        self,
        filepath: str,
        offset_strategy: str = "center"
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Ejecuta el pipeline completo de extracción para un archivo.

        Args:
            filepath: Ruta al archivo de audio (.wav, .mp3, .flac, .aif).
            offset_strategy: Estrategia de segmentación ("center" | "beginning").

        Returns:
            Diccionario con las matrices de características:
            {
                "mel_spectrogram": np.ndarray (128, T),
                "mfccs":           np.ndarray (60, T),
                "rms_energy":      np.ndarray (1, T),
                "filename":        str
            }
            Retorna None si el archivo no pudo procesarse.
        """
        filename = Path(filepath).name
        logger.info(f"→ Procesando: {filename}")

        # 1. Cargar y segmentar
        audio = self.load_audio_segment(filepath, offset_strategy)
        if audio is None:
            return None

        # 2. Extraer Mel Spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)

        # 3. Extraer MFCCs
        mfccs = self.extract_mfccs(audio)

        # 4. Extraer Energía RMS
        rms = self.extract_rms_energy(audio)

        return {
            "mel_spectrogram": mel_spec,   # (128, T)
            "mfccs":           mfccs,      # (60, T)
            "rms_energy":      rms,        # (1, T)
            "filename":        filename
        }


# ─────────────────────────────────────────────────────────────────────
# FUNCIÓN DE DATASET: Procesar directorio completo de audio
# ─────────────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}


def collect_audio_files(directory: str) -> List[str]:
    """
    Recopila todos los archivos de audio soportados en un directorio.

    Args:
        directory: Ruta al directorio con los archivos de audio.

    Returns:
        Lista de rutas absolutas a los archivos encontrados.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_FORMATS:
                files.append(os.path.join(root, fname))
    files.sort()
    logger.info(f"  Encontrados {len(files)} archivos de audio en '{directory}'")
    return files


def extract_dataset_features(
    class1_dir: str,
    class0_dir: str,
    extractor: Optional[XsiveFeatureExtractor] = None
) -> Tuple[List[Dict], List[int]]:
    """
    Procesa los dos directorios de clases y retorna features + etiquetas.

    Args:
        class1_dir: Directorio con tracks Clase 1 (Vanguardista/Aprobado).
        class0_dir: Directorio con tracks Clase 0 (Antiguo/Denegado).
        extractor:  Instancia de XsiveFeatureExtractor (opcional; se crea si None).

    Returns:
        Tuple[features_list, labels_list]
        - features_list: Lista de dicts con matrices de características
        - labels_list:   Lista de enteros (0 o 1) correspondientes
    """
    if extractor is None:
        extractor = XsiveFeatureExtractor()

    features_list = []
    labels_list   = []

    # ── Clase 1: Vanguardista/Aprobado ──
    logger.info("━━━ Extrayendo Clase 1 (Vanguardista) ━━━")
    class1_files = collect_audio_files(class1_dir)
    for filepath in class1_files:
        feats = extractor.extract_all_features(filepath)
        if feats is not None:
            features_list.append(feats)
            labels_list.append(1)

    # ── Clase 0: Antiguo/Denegado ──
    logger.info("━━━ Extrayendo Clase 0 (Antiguo/Minimalista) ━━━")
    class0_files = collect_audio_files(class0_dir)
    for filepath in class0_files:
        feats = extractor.extract_all_features(filepath)
        if feats is not None:
            features_list.append(feats)
            labels_list.append(0)

    logger.info(
        f"\n✓ Extracción completa: {len(features_list)} tracks procesados "
        f"({labels_list.count(1)} Clase 1 | {labels_list.count(0)} Clase 0)"
    )
    return features_list, labels_list


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN STANDALONE (testing de extracción)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from config import CLASS1_DIR, CLASS0_DIR

    print("=" * 60)
    print("  XSIVE — Test de Extracción de Características")
    print("=" * 60)

    extractor = XsiveFeatureExtractor()

    # Test con un archivo si se pasa como argumento
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"\nProbando archivo: {test_file}")
        feats = extractor.extract_all_features(test_file)
        if feats:
            print(f"\n  Mel Spectrogram : {feats['mel_spectrogram'].shape}")
            print(f"  MFCCs (+ delta) : {feats['mfccs'].shape}")
            print(f"  RMS Energy      : {feats['rms_energy'].shape}")
    else:
        print(f"\nUso: python feature_extraction.py <ruta_audio>")
        print(f"O ejecuta main.py para procesar el dataset completo.")
