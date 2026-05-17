#!/usr/bin/env python3
"""
Dasha v2 — Систематический эксперимент по оценке разделимости векторов

Цель: проверить разные способы извлечения векторов на реальных данных

Параметры (меняй здесь):
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from pipeline import VoiceFeaturePipeline
from cv_ru_loader import load_speakers_with_audio

# ====================== НАСТРОЙКИ ЭКСПЕРИМЕНТА ======================
NUM_SPEAKERS = 50          # Начально 50, потом можно 500
NUM_ROUNDS = 3             # Начально 3, потом 5
PHRASES_PER_SPEAKER = 10   # Фраз на спикера
MIN_PHRASES = 10           # Минимум фраз у спикера

PIPELINE = VoiceFeaturePipeline(use_rasta=True)

# ====================== ЛОГИ ======================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
FULL_LOG = LOG_DIR / "experiment_full.log"
SUMMARY_LOG = LOG_DIR / "experiment_summary.log"


def log_full(msg: str):
    with open(FULL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    print(msg)

def log_summary(msg: str):
    with open(SUMMARY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")
    print(msg)


def compute_intra_metrics(vectors: List[np.ndarray]) -> Dict[str, float]:
    """MSE и RMSE между всеми парами векторов одного спикера (per dimension + overall)."""
    arr = np.array(vectors)  # (10, 78)
    n = len(arr)
    if n < 2:
        return {"mse_mean": 0.0, "rmse_mean": 0.0}

    mse_list = []
    for i in range(n):
        for j in range(i+1, n):
            diff = arr[i] - arr[j]
            mse = np.mean(diff ** 2)
            mse_list.append(mse)

    mse_mean = float(np.mean(mse_list))
    rmse_mean = float(np.sqrt(mse_mean))

    # Per-dimension MSE/RMSE
    per_dim_mse = np.mean([(arr[i] - arr[j])**2 for i in range(n) for j in range(i+1, n)], axis=0)
    per_dim_rmse = np.sqrt(per_dim_mse)

    return {
        "mse_mean": round(mse_mean, 6),
        "rmse_mean": round(rmse_mean, 6),
        "per_dim_mse_mean": round(float(np.mean(per_dim_mse)), 6),
        "per_dim_rmse_mean": round(float(np.mean(per_dim_rmse)), 6)
    }


def compute_inter_metrics(mean_vectors: List[np.ndarray]) -> Dict[str, float]:
    """Корреляция и cosine similarity между средними векторами разных спикеров."""
    arr = np.array(mean_vectors)
    n = len(arr)
    if n < 2:
        return {"mean_correlation": 0.0, "mean_cosine": 0.0}

    corrs = []
    cosines = []
    for i in range(n):
        for j in range(i+1, n):
            v1, v2 = arr[i], arr[j]
            corr = np.corrcoef(v1, v2)[0, 1]
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            corrs.append(corr)
            cosines.append(cosine)

    return {
        "mean_correlation": round(float(np.mean(corrs)), 4),
        "mean_cosine": round(float(np.mean(cosines)), 4),
        "std_correlation": round(float(np.std(corrs)), 4)
    }


def run_single_experiment(round_num: int, speakers_pool: Dict) -> Dict:
    """Один полный эксперимент (500 спикеров)."""
    log_full(f"\n=== Эксперимент #{round_num} ===")

    # 1. Выбираем 500 спикеров с минимум 10 фраз
    valid_speakers = [sid for sid, paths in speakers_pool.items() if len(paths) >= MIN_PHRASES]
    selected = random.sample(valid_speakers, min(NUM_SPEAKERS, len(valid_speakers)))

    speaker_results = []
    mean_vectors = []

    for s_idx, speaker_id in enumerate(selected):
        audio_paths = random.sample(speakers_pool[speaker_id], PHRASES_PER_SPEAKER)

        vectors = []
        for path in audio_paths:
            try:
                res = PIPELINE.extract_features(path)
                vec = np.array(res["normalized_vector"])
                vectors.append(vec)
            except Exception as e:
                log_full(f"  [WARN] {speaker_id}: {e}")
                continue

        if len(vectors) < 2:
            continue

        intra = compute_intra_metrics(vectors)
        mean_vec = np.mean(vectors, axis=0)
        mean_vectors.append(mean_vec)

        speaker_results.append({
            "speaker_id": speaker_id,
            "num_phrases": len(vectors),
            **intra
        })

        if (s_idx + 1) % 10 == 0:
            log_full(f"  Обработано {s_idx+1}/{len(selected)} спикеров...")

    inter = compute_inter_metrics(mean_vectors)

    result = {
        "round": round_num,
        "num_speakers": len(selected),
        "intra_avg_mse": round(float(np.mean([r["mse_mean"] for r in speaker_results])), 6),
        "intra_avg_rmse": round(float(np.mean([r["rmse_mean"] for r in speaker_results])), 6),
        **inter
    }

    log_full(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    log_full("=" * 60)
    log_full("Dasha v2 — Систематический эксперимент (v2.17)")
    log_full(f"Параметры: {NUM_SPEAKERS} спикеров, {NUM_ROUNDS} кругов, {PHRASES_PER_SPEAKER} фраз на спикера")
    log_full("=" * 60)

    # Загрузка датасета
    log_full("\n[Загрузка] Загружаем спикеров...")
    all_speakers = load_speakers_with_audio()
    log_full(f"[Загрузка] Всего спикеров: {len(all_speakers)}")

    all_results = []
    for r in range(1, NUM_ROUNDS + 1):
        res = run_single_experiment(r, all_speakers)
        all_results.append(res)

        log_summary(f"\n=== Эксперимент #{r} ===")
        log_summary(f"Спикеров: {res['num_speakers']}")
        log_summary(f"Intra MSE (avg): {res['intra_avg_mse']}")
        log_summary(f"Intra RMSE (avg): {res['intra_avg_rmse']}")
        log_summary(f"Inter Correlation (avg): {res['mean_correlation']}")
        log_summary(f"Inter Cosine (avg): {res['mean_cosine']}")

    # Общий вывод
    log_summary("\n" + "=" * 50)
    log_summary(" ОБЩИЙ ВЫВОД ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
    log_summary("=" * 50)
    log_summary(f"Средний Intra RMSE: {np.mean([r['intra_avg_rmse'] for r in all_results]):.6f}")
    log_summary(f"Средний Inter Correlation: {np.mean([r['mean_correlation'] for r in all_results]):.4f}")
    log_summary(f"Средний Inter Cosine: {np.mean([r['mean_cosine'] for r in all_results]):.4f}")
    log_summary("\nЭксперимент завершен. Смотри логи в logs/")

if __name__ == "__main__":
    main()