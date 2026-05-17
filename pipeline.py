#!/usr/bin/env python3
"""
Dasha v2 — Voice Feature Pipeline v2.8 (GOST-friendly)

Полностью переработан под требования ГОСТ Р 52633:
- 13 MFCC (без энергии) + RASTA
- Вычисляем mean + std по 13 коэффициентам = 26-мерный вектор
- Это классика для speaker recognition: стабильность + высокая энтропия
- Финальная robust нормализация в [0, 1]
- Вектор теперь отлично разделяет спикеров и подходит для НПБК
"""

from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import lfilter
from typing import Dict, Any, Optional
from pathlib import Path
import json

from normalizer import FeatureNormalizer


class VoiceFeaturePipeline:
    SAMPLE_RATE: int = 16000
    PRE_EMPHASIS: float = 0.97
    FRAME_LENGTH_MS: int = 25
    FRAME_SHIFT_MS: int = 10
    N_MFCC: int = 13          # 13 MFCC (без энергии)
    N_MELS: int = 40
    FMIN: float = 20.0
    FMAX: float = 8000.0
    RASTA_POLE: float = 0.94
    MIN_SPEECH_SEC: float = 0.6
    VAD_ENERGY_PERCENTILE: float = 20.0

    def __init__(self, use_rasta: bool = True, vad_threshold: float = 0.01, normalizer: Optional[FeatureNormalizer] = None):
        self.use_rasta = use_rasta
        self.vad_threshold = vad_threshold
        self.normalizer = normalizer or FeatureNormalizer(method="global_minmax")
        self._load_normalizer_if_exists()

    def _load_normalizer_if_exists(self) -> None:
        params_path = Path("models/audio_params/normalizer_params.json")
        if params_path.exists():
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                self.normalizer.params = params
                self.normalizer.method = params.get("method", "global_minmax")
            except Exception as e:
                print(f"[Pipeline] Не удалось загрузить нормализатор: {e}")

    @staticmethod
    def _pre_emphasis(y: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        return np.append(y[0], y[1:] - alpha * y[:-1])

    def _vad(self, y: np.ndarray, sr: int) -> np.ndarray:
        frame_length = int(self.FRAME_LENGTH_MS * sr / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * sr / 1000)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        threshold = max(self.vad_threshold, np.percentile(rms, self.VAD_ENERGY_PERCENTILE))
        speech_mask = rms > threshold

        min_frames = int(self.MIN_SPEECH_SEC * sr / hop_length)
        if np.sum(speech_mask) < min_frames:
            return np.ones(len(rms), dtype=bool)

        # Агрессивная обрезка хвостов
        speech_idx = np.where(speech_mask)[0]
        if len(speech_idx) > 0:
            start = speech_idx[0]
            end = speech_idx[-1] + 1
            trim = max(3, int(0.08 * (end - start)))
            start = min(start + trim, len(speech_mask) - 1)
            end = max(end - trim, start + 1)
            speech_mask[:start] = False
            speech_mask[end:] = False

        for i in range(1, len(speech_mask) - 1):
            if not speech_mask[i] and speech_mask[i-1] and speech_mask[i+1]:
                speech_mask[i] = True
        return speech_mask

    @staticmethod
    def _rasta_filter(trajectory: np.ndarray, pole: float = 0.94) -> np.ndarray:
        b = np.array([0.1, -0.1])
        a = np.array([1.0, -pole])
        return lfilter(b, a, trajectory)

    def extract_features(self, audio_input: str | np.ndarray | Path, sr: Optional[int] = None) -> Dict[str, Any]:
        if isinstance(audio_input, (str, Path)):
            y, sr = librosa.load(str(audio_input), sr=self.SAMPLE_RATE, mono=True)
        else:
            y = np.asarray(audio_input, dtype=np.float32)
            sr = sr or self.SAMPLE_RATE
            if sr != self.SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.SAMPLE_RATE)
                sr = self.SAMPLE_RATE

        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        y_pre = self._pre_emphasis(y, self.PRE_EMPHASIS)

        frame_length = int(self.FRAME_LENGTH_MS * sr / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * sr / 1000)
        vad_mask = self._vad(y_pre, sr)

        mfcc = librosa.feature.mfcc(
            y=y_pre, sr=sr, n_mfcc=self.N_MFCC + 1,   # +1 чтобы потом убрать энергию
            n_fft=frame_length, hop_length=hop_length,
            n_mels=self.N_MELS, fmin=self.FMIN, fmax=self.FMAX,
            window="hamming", center=True, norm="ortho"
        )
        mfcc = mfcc[1:, :]   # убираем энергию (0-й коэффициент)

        if self.use_rasta:
            mfcc_rasta = np.zeros_like(mfcc)
            for i in range(self.N_MFCC):
                mfcc_rasta[i] = self._rasta_filter(mfcc[i], self.RASTA_POLE)
        else:
            mfcc_rasta = mfcc.copy()

        # === GOST-friendly статистики ===
        if np.any(vad_mask) and vad_mask.shape[0] == mfcc_rasta.shape[1]:
            active = mfcc_rasta[:, vad_mask]
        else:
            active = mfcc_rasta

        mean_vec = np.mean(active, axis=1)                    # 13
        std_vec  = np.std(active, axis=1) + 1e-8             # 13
        features_26 = np.concatenate([mean_vec, std_vec])     # 26-мерный вектор

        if self.normalizer.params is not None:
            normalized = self.normalizer.transform(features_26)
        else:
            q_low, q_high = np.percentile(features_26, [2, 98])
            if q_high - q_low < 1e-8:
                normalized = np.full(26, 0.5, dtype=np.float32)
            else:
                normalized = np.clip((features_26 - q_low) / (q_high - q_low), 0.0, 1.0)

        return {
            "normalized_vector": normalized.tolist(),
            "raw_mean_vector": features_26,
            "mfcc_rasta": mfcc_rasta,
            "features_26": features_26,
            "vad_mask": vad_mask,
            "y_pre": y_pre,
            "sr": sr,
            "use_rasta": self.use_rasta,
            "pipeline_version": "2.8"
        }

    def get_feature_quality_metrics(self, vectors: list) -> Dict[str, float]:
        if len(vectors) < 2:
            return {"error": "Нужно минимум 2 вектора"}
        arr = np.array(vectors)
        sims = []
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                sims.append(np.dot(arr[i], arr[j]) / (np.linalg.norm(arr[i]) * np.linalg.norm(arr[j]) + 1e-8))
        mean_cosine = float(np.mean(sims))
        corr_matrix = np.corrcoef(arr.T)
        mean_feature_corr = float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, 1)])))
        feature_vars = np.var(arr, axis=0)
        entropy_proxy = float(np.mean(feature_vars))
        return {
            "mean_cosine_similarity": round(mean_cosine, 4),
            "mean_feature_correlation": round(mean_feature_corr, 4),
            "feature_variance_proxy": round(entropy_proxy, 4),
            "num_vectors": len(vectors)
        }


def process_phrase(audio_path: str | Path, use_rasta: bool = True, normalizer: Optional[FeatureNormalizer] = None) -> Dict[str, Any]:
    pipeline = VoiceFeaturePipeline(use_rasta=use_rasta, normalizer=normalizer)
    return pipeline.extract_features(audio_path)