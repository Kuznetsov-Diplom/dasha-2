#!/usr/bin/env python3
"""
Dasha v2.24 — Систематический эксперимент (RASTA=OFF)

Сравнение 26-dim vs 39-dim БЕЗ RASTA
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from pipeline import VoiceFeaturePipeline
from cv_ru_loader import load_speakers_with_audio
from normalizer import FeatureNormalizer

# ====================== НАСТРОЙКИ ======================
NUM_SPEAKERS = 300
NUM_ROUNDS = 3
PHRASES_PER_SPEAKER = 8
MIN_PHRASES = 8

# ====================== ВАРИАНТЫ (без RASTA) ======================
VARIANTS = {
    "26dim_no_rasta": {
        "name": "26-dim (MFCC only, RASTA=OFF)",
        "dim": 26,
        "use_deltas": False,
        "use_cmvn": True
    },
    "39dim_no_rasta": {
        "name": "39-dim (MFCC+Δ+ΔΔ, RASTA=OFF)",
        "dim": 39,
        "use_deltas": True,
        "use_cmvn": True
    }
}

PIPELINE = VoiceFeaturePipeline(use_rasta=False)  # ГЛАВНОЕ ИЗМЕНЕНИЕ
NORMALIZER = FeatureNormalizer(method="standard")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
FULL_LOG = LOG_DIR / "experiment_v2.24_no_rasta.log"


def log_full(msg: str):
    with open(FULL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)

def extract_vector(audio_path: str, variant: dict) -> np.ndarray:
    res = PIPELINE.extract_features(audio_path)
    vec = np.array(res["raw_mean_vector"])
    if not variant["use_deltas"]:
        vec = vec[:26]
    return vec

def compute_intra_rmse(vectors: List[np.ndarray]) -> float:
    arr = np.array(vectors)
    n = len(arr)
    if n < 2:
        return 0.0
    diffs = []
    for i in range(n):
        for j in range(i+1, n):
            diffs.append(np.mean((arr[i] - arr[j]) ** 2))
    return float(np.sqrt(np.mean(diffs)))

def compute_inter_correlation(mean_vectors: List[np.ndarray]) -> float:
    arr = np.array(mean_vectors)
    n = len(arr)
    if n < 2:
        return 0.0
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            corrs.append(np.corrcoef(arr[i], arr[j])[0, 1])
    return float(np.mean(corrs))

def run_variant(variant_key: str, variant: dict, speakers_pool: Dict, round_num: int) -> Dict:
    log_full(f"\n--- {variant['name']} (round {round_num}) ---")

    valid = [sid for sid, p in speakers_pool.items() if len(p) >= MIN_PHRASES]
    selected = random.sample(valid, min(NUM_SPEAKERS, len(valid)))

    intra_scores = []
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
        intra_scores.append(compute_intra_rmse(vectors))
        mean_vectors.append(np.mean(vectors, axis=0))

        if (s_idx + 1) % 50 == 0:
            log_full(f"  {s_idx+1}/{len(selected)} ...")

    inter_corr = compute_inter_correlation(mean_vectors)
    avg_intra = float(np.mean(intra_scores))

    result = {
        "variant": variant_key,
        "round": round_num,
        "num_speakers": len(selected),
        "intra_rmse": round(avg_intra, 6),
        "inter_correlation": round(inter_corr, 4)
    }
    log_full(json.dumps(result, ensure_ascii=False))
    return result

def main():
    log_full("=" * 70)
    log_full("Dasha v2.24 — Сравнение 26-dim vs 39-dim БЕЗ RASTA")
    log_full(f"Варианты: {list(VARIANTS.keys())}")
    log_full("=" * 70)

    all_speakers = load_speakers_with_audio()
    log_full(f"\n[Загрузка] {len(all_speakers)} спикеров")

    for round_num in range(1, NUM_ROUNDS + 1):
        log_full(f"\n========== КРУГ {round_num} / {NUM_ROUNDS} ==========")
        for vkey, vcfg in VARIANTS.items():
            run_variant(vkey, vcfg, all_speakers, round_num)

    log_full("\nЭксперимент завершён. Смотри лог: logs/experiment_v2.24_no_rasta.log")

if __name__ == "__main__":
    main()