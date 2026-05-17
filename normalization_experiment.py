#!/usr/bin/env python3
"""
Dasha v2.27 — Эксперимент по нормализации 13-мерного вектора

Цель: найти лучший метод приведения каждого из 13 признаков к [0,1]
с максимальным сохранением корреляционной структуры спикеров.

Методы:
1. No global (только CMVN)
2. Global Standard
3. Global MinMax (абсолютный)
4. Global MinMax (5-95 перцентиль)
5. Robust Scaling (IQR)
6. Quantile Transform (CDF)

Вывод: средние метрики + корреляция "спикер vs спикер" до и после.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import random

from pipeline import VoiceFeaturePipeline
from cv_ru_loader import load_speakers_with_audio

# ====================== НАСТРОЙКИ ======================
NUM_SPEAKERS = 400
NUM_ROUNDS = 3
PHRASES_PER_SPEAKER = 8

METHODS = [
    "no_global",
    "global_standard",
    "global_minmax_abs",
    "global_minmax_5_95",
    "robust_iqr",
    "quantile_cdf"
]

pipeline = VoiceFeaturePipeline(use_rasta=False, use_deltas=False, use_std=False)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "normalization_experiment_v2.27.log"


def log(msg: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)


def get_raw_vectors(speaker_paths: List[str]) -> np.ndarray:
    vecs = []
    for p in speaker_paths:
        try:
            res = pipeline.extract_features(p)
            vecs.append(res["features"])   # 13-dim mean only
        except:
            continue
    return np.array(vecs) if vecs else np.array([])


def compute_speaker_stats(vectors: np.ndarray) -> Dict:
    if len(vectors) < 2:
        return {}
    arr = np.array(vectors)
    intra_corrs = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            intra_corrs.append(np.corrcoef(arr[i], arr[j])[0, 1])
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "intra_corr_mean": float(np.mean(intra_corrs)),
        "intra_corr_std": float(np.std(intra_corrs))
    }


def apply_normalization(vectors: np.ndarray, method: str, params: Dict = None) -> np.ndarray:
    arr = np.array(vectors, dtype=np.float32)
    if method == "no_global":
        return arr
    elif method == "global_standard":
        mean = np.array(params["mean"])
        std = np.array(params["std"]) + 1e-8
        return (arr - mean) / std
    elif method == "global_minmax_abs":
        minv = np.array(params["min"])
        maxv = np.array(params["max"])
        return (arr - minv) / (maxv - minv + 1e-8)
    elif method == "global_minmax_5_95":
        minv = np.array(params["p5"])
        maxv = np.array(params["p95"])
        return np.clip((arr - minv) / (maxv - minv + 1e-8), 0, 1)
    elif method == "robust_iqr":
        median = np.array(params["median"])
        iqr = np.array(params["iqr"]) + 1e-8
        return (arr - median) / iqr
    elif method == "quantile_cdf":
        # Простая реализация через ранги (для эксперимента)
        ranked = np.argsort(np.argsort(arr, axis=0), axis=0)
        return ranked / (len(arr) - 1)
    return arr


def get_normalization_params(all_vectors: List[np.ndarray], method: str) -> Dict:
    arr = np.concatenate(all_vectors, axis=0)
    if method == "global_standard":
        return {"mean": np.mean(arr, axis=0).tolist(), "std": np.std(arr, axis=0).tolist()}
    elif method == "global_minmax_abs":
        return {"min": np.min(arr, axis=0).tolist(), "max": np.max(arr, axis=0).tolist()}
    elif method == "global_minmax_5_95":
        return {"p5": np.percentile(arr, 5, axis=0).tolist(), "p95": np.percentile(arr, 95, axis=0).tolist()}
    elif method == "robust_iqr":
        return {"median": np.median(arr, axis=0).tolist(), "iqr": (np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)).tolist()}
    return {}


def run_round(round_num: int, all_speakers: Dict):
    log(f"\n========== КРУГ {round_num} / {NUM_ROUNDS} ==========")

    valid_speakers = [sid for sid, paths in all_speakers.items() if len(paths) >= PHRASES_PER_SPEAKER]
    selected = random.sample(valid_speakers, min(NUM_SPEAKERS, len(valid_speakers)))

    speaker_raw_stats = []
    all_raw_vectors = []

    for sid in selected:
        paths = random.sample(all_speakers[sid], PHRASES_PER_SPEAKER)
        vecs = get_raw_vectors(paths)
        if len(vecs) < 2:
            continue
        stats = compute_speaker_stats(vecs)
        speaker_raw_stats.append(stats)
        all_raw_vectors.append(vecs)

    # Обучаем параметры нормализации на всех данных
    params = {}
    for method in METHODS:
        params[method] = get_normalization_params(all_raw_vectors, method)

    results = {}
    for method in METHODS:
        normalized_speaker_stats = []
        all_normalized = []
        for vecs in all_raw_vectors:
            norm_vecs = apply_normalization(vecs, method, params[method])
            stats = compute_speaker_stats(norm_vecs)
            normalized_speaker_stats.append(stats)
            all_normalized.append(norm_vecs)

        # Общая корреляция спикер vs спикер (средние векторы)
        mean_raw = [np.mean(v, axis=0) for v in all_raw_vectors]
        mean_norm = [np.mean(v, axis=0) for v in all_normalized]

        inter_raw = []
        inter_norm = []
        for i in range(len(mean_raw)):
            for j in range(i+1, len(mean_raw)):
                inter_raw.append(np.corrcoef(mean_raw[i], mean_raw[j])[0, 1])
                inter_norm.append(np.corrcoef(mean_norm[i], mean_norm[j])[0, 1])

        results[method] = {
            "intra_corr_before": np.mean([s["intra_corr_mean"] for s in speaker_raw_stats]),
            "intra_corr_after": np.mean([s["intra_corr_mean"] for s in normalized_speaker_stats]),
            "inter_corr_before": np.mean(inter_raw),
            "inter_corr_after": np.mean(inter_norm),
            "range_after_min": float(np.min([np.min(v) for v in all_normalized])),
            "range_after_max": float(np.max([np.max(v) for v in all_normalized])),
        }

    # Вывод
    log("\nМетод                    | Intra до/после | Inter до/после | Диапазон после")
    log("-" * 75)
    for m in METHODS:
        r = results[m]
        log(f"{m:25} | {r['intra_corr_before']:.4f} / {r['intra_corr_after']:.4f} | {r['inter_corr_before']:.4f} / {r['inter_corr_after']:.4f} | [{r['range_after_min']:.2f}, {r['range_after_max']:.2f}]")

    return results

def main():
    log("=" * 80)
    log("Dasha v2.27 — Эксперимент по нормализации 13-dim (400 спикеров × 3 круга)")
    log("Методы: " + ", ".join(METHODS))
    log("=" * 80)

    all_speakers = load_speakers_with_audio()
    log(f"Загружено {len(all_speakers)} спикеров")

    all_results = {m: [] for m in METHODS}

    for r in range(1, NUM_ROUNDS + 1):
        round_res = run_round(r, all_speakers)
        for m in METHODS:
            all_results[m].append(round_res[m])

    # Итоговый вывод
    log("\n" + "=" * 80)
    log("ИТОГОВЫЕ СРЕДНИЕ РЕЗУЛЬТАТЫ")
    log("=" * 80)
    for m in METHODS:
        res_list = all_results[m]
        avg_intra = np.mean([r["intra_corr_after"] for r in res_list])
        avg_inter = np.mean([r["inter_corr_after"] for r in res_list])
        log(f"{m:25} | Intra={avg_intra:.4f} | Inter={avg_inter:.4f}")

    log("\nЭксперимент завершён. Лог: " + str(LOG_FILE))

if __name__ == "__main__":
    main()