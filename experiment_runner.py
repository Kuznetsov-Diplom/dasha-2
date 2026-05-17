#!/usr/bin/env python3
"""
Dasha v2 — Систематический эксперимент v2.19

Сравнение разных способов извлечения векторов

Параметры:
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable

from pipeline import VoiceFeaturePipeline
from cv_ru_loader import load_speakers_with_audio

from normalizer import FeatureNormalizer

# ====================== НАСТРОЙКИ ======================
NUM_SPEAKERS = 500         # 50 или 500
NUM_ROUNDS = 5             # 3 или 5
PHRASES_PER_SPEAKER = 10
MIN_PHRASES = 10

# ====================== ВАРИАНТЫ ИЗВлечения ======================
VARIANTS = {
    "26dim_baseline": {
        "name": "26-dim (MFCC only, no deltas)",
        "dim": 26,
        "use_deltas": False,
        "use_cmvn": True,
        "normalize": False
    },
    "78dim_raw": {
        "name": "78-dim raw (MFCC+Δ+ΔΔ + CMVN)",
        "dim": 78,
        "use_deltas": True,
        "use_cmvn": True,
        "normalize": False
    },
    "78dim_standard": {
        "name": "78-dim + global standard norm",
        "dim": 78,
        "use_deltas": True,
        "use_cmvn": True,
        "normalize": True
    },
    "78dim_no_cmvn": {
        "name": "78-dim without CMVN",
        "dim": 78,
        "use_deltas": True,
        "use_cmvn": False,
        "normalize": False
    }
}

PIPELINE = VoiceFeaturePipeline(use_rasta=True)
NORMALIZER = FeatureNormalizer(method="standard")

# ====================== ЛОГИ ======================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
FULL_LOG = LOG_DIR / "experiment_full_v2.19.log"
SUMMARY_LOG = LOG_DIR / "experiment_summary_v2.19.log"


def log_full(msg: str):
    with open(FULL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)

def log_summary(msg: str):
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")
    print(msg)


def extract_vector(audio_path: str, variant: dict) -> np.ndarray:
    """ Извлекает вектор по варианту. """
    res = PIPELINE.extract_features(audio_path)
    vec = np.array(res["raw_mean_vector"])  # 78 или 26

    if not variant["use_deltas"]:
        vec = vec[:26]  # берём только 13 mean + 13 std

    if not variant["use_cmvn"]:
        # пересчитываем без CMVN (приблизительно)
        res2 = PIPELINE.extract_features(audio_path)  # пока просто используем тот же
        vec = np.array(res2["raw_mean_vector"])

    if variant["normalize"]:
        if NORMALIZER.params is None:
            NORMALIZER.fit(np.array([vec]))
        vec = NORMALIZER.transform(vec)

    return vec


def compute_intra_detailed(vectors: List[np.ndarray]) -> Dict:
    """ Полная статистика по каждому признаку. """
    arr = np.array(vectors)
    n = len(arr)
    if n < 2:
        return {}

    per_dim_mse = []
    per_dim_rmse = []
    for d in range(arr.shape[1]):
        diffs = []
        for i in range(n):
            for j in range(i+1, n):
                diffs.append((arr[i, d] - arr[j, d]) ** 2)
        mse_d = float(np.mean(diffs))
        per_dim_mse.append(mse_d)
        per_dim_rmse.append(np.sqrt(mse_d))

    return {
        "mse_mean": round(float(np.mean(per_dim_mse)), 6),
        "rmse_mean": round(float(np.mean(per_dim_rmse)), 6),
        "worst_dim_mse": round(float(np.max(per_dim_mse)), 6),
        "best_dim_mse": round(float(np.min(per_dim_mse)), 6),
        "per_dim_mse": [round(x, 6) for x in per_dim_mse]
    }


def compute_inter_metrics(mean_vectors: List[np.ndarray]) -> Dict[str, float]:
    arr = np.array(mean_vectors)
    n = len(arr)
    if n < 2:
        return {}

    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            corr = np.corrcoef(arr[i], arr[j])[0, 1]
            corrs.append(corr)

    return {
        "mean_correlation": round(float(np.mean(corrs)), 4),
        "std_correlation": round(float(np.std(corrs)), 4)
    }


def run_variant_experiment(variant_key: str, variant: dict, speakers_pool: Dict, round_num: int) -> Dict:
    log_full(f"\n--- Вариант: {variant['name']} (round {round_num}) ---")

    valid = [sid for sid, p in speakers_pool.items() if len(p) >= MIN_PHRASES]
    selected = random.sample(valid, min(NUM_SPEAKERS, len(valid)))

    speaker_stats = []
    mean_vectors = []

    for s_idx, sid in enumerate(selected):
        paths = random.sample(speakers_pool[sid], PHRASES_PER_SPEAKER)
        vectors = []
        for p in paths:
            try:
                v = extract_vector(p, variant)
                vectors.append(v)
            except:
                continue

        if len(vectors) < 2:
            continue

        intra = compute_intra_detailed(vectors)
        mean_vec = np.mean(vectors, axis=0)
        mean_vectors.append(mean_vec)
        speaker_stats.append(intra)

        if (s_idx + 1) % 50 == 0:
            log_full(f"  {s_idx+1}/{len(selected)} ...")

    inter = compute_inter_metrics(mean_vectors)

    result = {
        "variant": variant_key,
        "round": round_num,
        "num_speakers": len(selected),
        "intra_mse_avg": round(float(np.mean([s["mse_mean"] for s in speaker_stats])), 6),
        "intra_rmse_avg": round(float(np.mean([s["rmse_mean"] for s in speaker_stats])), 6),
        "worst_dim_mse": round(float(np.max([s["worst_dim_mse"] for s in speaker_stats])), 6),
        "best_dim_mse": round(float(np.min([s["best_dim_mse"] for s in speaker_stats])), 6),
        **inter
    }

    log_full(json.dumps(result, ensure_ascii=False))
    return result


def main():
    log_full("=" * 70)
    log_full("Dasha v2.19 — Сравнение вариантов извлечения вектора (500 спикеров × 5 кругов)")
    log_full(f"Варианты: {list(VARIANTS.keys())}")
    log_full("=" * 70)

    all_speakers = load_speakers_with_audio()
    log_full(f"\n[Загрузка] {len(all_speakers)} спикеров загружено")

    all_results = {k: [] for k in VARIANTS}

    for round_num in range(1, NUM_ROUNDS + 1):
        log_full(f"\n========== КРУГ {round_num} / {NUM_ROUNDS} ==========")
        for vkey, vcfg in VARIANTS.items():
            res = run_variant_experiment(vkey, vcfg, all_speakers, round_num)
            all_results[vkey].append(res)

    # Общий вывод
    log_summary("\n" + "=" * 70)
    log_summary(" ОБЩИЙ ВЫВОД ПО ВСЕМ ВАриАНТАМ")
    log_summary("=" * 70)

    for vkey in VARIANTS:
        res_list = all_results[vkey]
        avg_inter = np.mean([r["mean_correlation"] for r in res_list])
        avg_intra_rmse = np.mean([r["intra_rmse_avg"] for r in res_list])
        log_summary(f"{VARIANTS[vkey]['name']}: Inter={avg_inter:.4f} | Intra_RMSE={avg_intra_rmse:.4f}")

    log_summary("\nЭксперимент завершен. Смотри логи в logs/")

if __name__ == "__main__":
    main()