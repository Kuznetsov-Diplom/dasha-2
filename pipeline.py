#!/usr/bin/env python3
"""
Dasha v2.41 — pipeline + подробная отладка обработки звука

Добавлено:
- get_audio_debug_info() — полный разбор одного файла
  (VAD %, MFCC статистика до/после CMVN, raw vs norm vector)
- Используется в app.py для показа "🔍 Отладка обработки звука"
"""

from __future__ import annotations
import numpy as np
import librosa
import scipy.signal as signal
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from normalizer import FeatureNormalizer

class VoiceFeaturePipeline:
    SAMPLE_RATE: int = 16000
    PRE_EMPHASIS: float = 0.97
    FRAME_LENGTH_MS: int = 25
    FRAME_SHIFT_MS: int = 10
    N_MFCC: int = 13
    N_MELS: int = 40
    FMIN: float = 20.0
    FMAX: float = 8000.0
    MIN_SPEECH_SEC: float = 0.6
    VAD_ENERGY_PERCENTILE: float = 20.0

    def __init__(self, use_rasta: bool = False, use_deltas: bool = False, normalizer: Optional[FeatureNormalizer] = None):
        self.use_rasta = use_rasta
        self.use_deltas = use_deltas
        self.normalizer = normalizer or FeatureNormalizer(method="global_minmax_abs")
        self._load_normalizer_if_exists()

    def _load_normalizer_if_exists(self) -> None:
        params_path = Path("models/audio_params/normalizer_params.json")
        if params_path.exists():
            try:
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                self.normalizer.params = params
                self.normalizer.method = params.get("method", "global_minmax_abs")
            except Exception as e:
                print(f"[Pipeline] Не удалось загрузить нормализатор: {e}")

    @staticmethod
    def _pre_emphasis(y: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        return np.append(y[0], y[1:] - alpha * y[:-1])

    def _vad(self, y: np.ndarray, sr: int) -> np.ndarray:
        frame_length = int(self.FRAME_LENGTH_MS * sr / 1000)
        hop_length = int(self.FRAME_SHIFT_MS * sr / 1000)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        threshold = max(0.01, np.percentile(rms, self.VAD_ENERGY_PERCENTILE))
        speech_mask = rms > threshold

        min_frames = int(self.MIN_SPEECH_SEC * sr / hop_length)
        if np.sum(speech_mask) < min_frames:
            return np.ones(len(rms), dtype=bool)

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
    def _cmvn(mfcc: np.ndarray, window: int = 301) -> np.ndarray:
        mfcc = mfcc.astype(np.float32)
        n_coeffs, n_frames = mfcc.shape
        if n_frames < window:
            window = n_frames
        mfcc_norm = np.zeros_like(mfcc)
        for c in range(n_coeffs):
            feat = mfcc[c]
            local_mean = np.convolve(feat, np.ones(window) / window, mode='same')
            local_std = np.convolve(np.abs(feat - local_mean), np.ones(window) / window, mode='same') + 1e-8
            mfcc_norm[c] = (feat - local_mean) / local_std
        return mfcc_norm

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
            y=y_pre, sr=sr, n_mfcc=self.N_MFCC,
            n_fft=frame_length, hop_length=hop_length,
            n_mels=self.N_MELS, fmin=self.FMIN, fmax=self.FMAX,
            window="hamming", center=True, norm="ortho"
        )

        mfcc_norm = self._cmvn(mfcc)

        if np.any(vad_mask) and vad_mask.shape[0] == mfcc_norm.shape[1]:
            active = mfcc_norm[:, vad_mask]
        else:
            active = mfcc_norm

        mean_vec = np.mean(active, axis=1)

        full_vec = mean_vec
        dim = 13
        dim_label = "13-dim (mean only) + global_minmax_abs (ГОСТ-ready)"

        normalized = self.normalizer.transform(full_vec)

        return {
            "normalized_vector": normalized.tolist(),
            "raw_mean_vector": full_vec.tolist(),
            "mfcc_norm": mfcc_norm,
            "features": full_vec,
            "vad_mask": vad_mask,
            "y_pre": y_pre,
            "sr": sr,
            "dim": dim,
            "dim_label": dim_label,
            "pipeline_version": "v2.42"
        }

    def get_audio_debug_info(self, audio_input: str | Path) -> Dict[str, Any]:
        """Полная отладка обработки одного аудио-файла"""
        try:
            res = self.extract_features(audio_input)
            y = res["y_pre"]
            vad_mask = res["vad_mask"]
            mfcc = res["mfcc_norm"]
            active = mfcc[:, vad_mask] if np.any(vad_mask) else mfcc

            speech_frames = int(np.sum(vad_mask))
            total_frames = len(vad_mask)
            speech_percent = round(100 * speech_frames / max(1, total_frames), 1)

            mfcc_mean = np.mean(active, axis=1).round(4).tolist()
            mfcc_std = np.std(active, axis=1).round(4).tolist()

            return {
                "filename": str(audio_input),
                "duration_sec": round(len(y) / self.SAMPLE_RATE, 2),
                "speech_frames_percent": speech_percent,
                "speech_frames": speech_frames,
                "total_frames": total_frames,
                "mfcc_mean_1_13": mfcc_mean,
                "mfcc_std_1_13": mfcc_std,
                "normalized_vector_preview": res["normalized_vector"][:5] + ["..."],
                "raw_mean_preview": res["raw_mean_vector"][:5] + ["..."],
                "vad_ok": speech_percent > 40
            }
        except Exception as e:
            return {"error": str(e), "filename": str(audio_input)}

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
        return {
            "mean_cosine_similarity": round(mean_cosine, 4),
            "mean_feature_correlation": round(mean_feature_corr, 4),
            "num_vectors": len(vectors)
        }

    def compute_eer(self, own_scores: list, alien_scores: list) -> float:
        """Простая оценка EER (ошибка 1 и 2 рода)"""
        from sklearn.metrics import roc_curve
        y_true = [1]*len(own_scores) + [0]*len(alien_scores)
        y_scores = own_scores + alien_scores
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        return round(eer, 4)