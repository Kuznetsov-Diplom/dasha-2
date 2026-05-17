#!/usr/bin/env python3
"""
FeatureNormalizer v2.27 — глобальная нормализация 13-мерных векторов

Поддерживаемые методы:
- global_minmax_abs (рекомендуется для ГОСТ — строго [0,1])
- standard (для экспериментов)

Все параметры сохраняются в JSON для reproducibility.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any


class FeatureNormalizer:
    def __init__(self, method: str = "global_minmax_abs"):
        self.method = method
        self.params: Optional[Dict[str, list]] = None

    def fit(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != 13:
            raise ValueError(f"Ожидается массив (N, 13), получено {vectors.shape}")

        if self.method == "global_minmax_abs":
            self.params = {
                "method": "global_minmax_abs",
                "min": np.min(vectors, axis=0).tolist(),
                "max": np.max(vectors, axis=0).tolist()
            }
        elif self.method == "standard":
            self.params = {
                "method": "standard",
                "mean": np.mean(vectors, axis=0).tolist(),
                "std": (np.std(vectors, axis=0) + 1e-8).tolist()
            }
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

    def transform(self, vector: np.ndarray) -> np.ndarray:
        if self.params is None:
            # fallback
            v_min, v_max = np.min(vector), np.max(vector)
            if v_max - v_min < 1e-8:
                return np.full_like(vector, 0.5)
            return (vector - v_min) / (v_max - v_min)

        vec = np.asarray(vector, dtype=np.float32)

        if self.method == "global_minmax_abs":
            minv = np.array(self.params["min"], dtype=np.float32)
            maxv = np.array(self.params["max"], dtype=np.float32)
            scaled = (vec - minv) / (maxv - minv + 1e-8)
            return np.clip(scaled, 0.0, 1.0)

        elif self.method == "standard":
            mean = np.array(self.params["mean"], dtype=np.float32)
            std = np.array(self.params["std"], dtype=np.float32)
            return np.clip((vec - mean) / std + 0.5, 0, 1)

        return vec

    def save(self, path: str | Path = "models/audio_params/normalizer_params.json") -> None:
        if self.params is None:
            raise RuntimeError("Нормализатор не обучен")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)
        print(f"[Normalizer] Параметры сохранены: {p}")

    def load(self, path: str | Path = "models/audio_params/normalizer_params.json") -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "r", encoding="utf-8") as f:
            self.params = json.load(f)
        self.method = self.params.get("method", "global_minmax_abs")
        return True

    def is_fitted(self) -> bool:
        return self.params is not None