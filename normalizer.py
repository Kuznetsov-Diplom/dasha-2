#!/usr/bin/env python3
"""
FeatureNormalizer v2.5 — глобальная нормализация 13-мерных векторов (ГОСТ Р 52633.5)

Поддерживаемые методы:
  - global_minmax_abs  ← РЕКОМЕНДУЕТСЯ по ГОСТ (строго [0, 1])
  - standard           ← только для экспериментов

Формула global_minmax_abs:
  v_norm[i] = clip((v[i] - min[i]) / (max[i] - min[i] + ε), 0, 1)

Параметры сохраняются в JSON для воспроизводимости.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List


class FeatureNormalizer:
    def __init__(self, method: str = "global_minmax_abs"):
        self.method = method
        self.params: Optional[Dict] = None

    # ── Обучение ─────────────────────────────────────────────────────────────

    def fit(self, vectors: np.ndarray, pipeline_tag: Optional[str] = None) -> None:
        """
        Обучить нормализатор на массиве (N, 13).
        По ГОСТ Р 52633.5 рекомендуется global_minmax_abs.

        Используются 1й и 99й перцентили вместо абсолютных min/max
        (робастная нормализация): после отключения CMVN raw MFCC могут
        давать редкие выбросы, которые иначе съели бы весь диапазон.
        """
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 13:
            raise ValueError(f"Ожидается (N, 13), получено {arr.shape}")
        if len(arr) < 10:
            raise ValueError(f"Слишком мало векторов для обучения: {len(arr)}. Нужно ≥ 10.")

        if self.method == "global_minmax_abs":
            min_v = np.percentile(arr, 1, axis=0)
            max_v = np.percentile(arr, 99, axis=0)
            degenerate = np.where((max_v - min_v) < 1e-6)[0]
            if len(degenerate) > 0:
                print(f"[Normalizer] ⚠️ Вырожденные признаки (диапазон ≈ 0): F{degenerate + 1}")
            self.params = {
                "method": "global_minmax_abs",
                "min": min_v.tolist(),
                "max": max_v.tolist(),
                "n_samples": int(len(arr)),
                "coverage_pct": self._compute_coverage(arr, min_v, max_v),
                "pipeline_tag": pipeline_tag,
                "robust": "percentile_1_99",
            }

        elif self.method == "standard":
            # Только для экспериментов — НЕ используется в НПБК
            self.params = {
                "method": "standard",
                "mean": np.mean(arr, axis=0).tolist(),
                "std": (np.std(arr, axis=0) + 1e-8).tolist(),
                "n_samples": int(len(arr))
            }
        else:
            raise ValueError(f"Неизвестный метод: {self.method}. Используйте 'global_minmax_abs'.")

    def fit_from_list(self, vectors: List[List[float]]) -> None:
        """Обучить из списка векторов (удобный вариант)."""
        self.fit(np.array(vectors, dtype=np.float64))

    # ── Применение ───────────────────────────────────────────────────────────

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Нормализовать один вектор. Если параметры не загружены — fallback."""
        vec = np.asarray(vector, dtype=np.float32)

        if self.params is None:
            # Fallback: локальная нормализация (менее точная)
            v_min, v_max = np.min(vec), np.max(vec)
            if v_max - v_min < 1e-8:
                return np.full_like(vec, 0.5)
            return np.clip((vec - v_min) / (v_max - v_min), 0.0, 1.0)

        method = self.params.get("method", "global_minmax_abs")

        if method == "global_minmax_abs":
            min_v = np.array(self.params["min"], dtype=np.float32)
            max_v = np.array(self.params["max"], dtype=np.float32)
            scaled = (vec - min_v) / (max_v - min_v + 1e-8)
            return np.clip(scaled, 0.0, 1.0)

        elif method == "standard":
            mean = np.array(self.params["mean"], dtype=np.float32)
            std = np.array(self.params["std"], dtype=np.float32)
            # Центрируем и клипуем в [0, 1] через сдвиг
            return np.clip((vec - mean) / std * 0.2 + 0.5, 0.0, 1.0)

        return vec

    def transform_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Нормализовать массив (N, 13)."""
        return np.array([self.transform(v) for v in vectors])

    # ── Сохранение / загрузка ─────────────────────────────────────────────────

    def save(self, path: str = "models/audio_params/normalizer_params.json") -> None:
        if self.params is None:
            raise RuntimeError("Нормализатор не обучен. Вызовите fit() сначала.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)
        print(f"[Normalizer] ✅ Параметры сохранены: {p} ({self.params.get('n_samples', '?')} векторов)")

    def load(self, path: str = "models/audio_params/normalizer_params.json") -> bool:
        p = Path(path)
        if not p.exists():
            print(f"[Normalizer] ℹ️ Файл не найден: {p}. Будет использован fallback.")
            return False
        with open(p, "r", encoding="utf-8") as f:
            self.params = json.load(f)
        self.method = self.params.get("method", "global_minmax_abs")
        n = self.params.get("n_samples", "?")
        print(f"[Normalizer] ✅ Параметры загружены: {p} (method={self.method}, n={n})")
        return True

    # ── Статус и диагностика ─────────────────────────────────────────────────

    def is_fitted(self) -> bool:
        return self.params is not None

    def get_info(self) -> Dict:
        """Вернуть информацию о текущих параметрах."""
        if not self.is_fitted():
            return {"status": "не обучен", "method": self.method}
        return {
            "status": "обучен",
            "method": self.params.get("method"),
            "n_samples": self.params.get("n_samples", "?"),
            "coverage_pct": self.params.get("coverage_pct", "?")
        }

    def _compute_coverage(self, arr: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> float:
        """Вычислить % значений, которые попадут в [0,1] после нормализации."""
        scaled = (arr - min_v) / (max_v - min_v + 1e-8)
        in_range = np.sum((scaled >= 0) & (scaled <= 1))
        return round(float(in_range / scaled.size * 100), 2)