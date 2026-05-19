#!/usr/bin/env python3
"""
analyze_separability.py — сравнение СПОСОБОВ предобработки голоса
по их разделимости спикеров на датасете Common Voice RU.

Сценарий:
  1. По каждому файлу ОДИН раз извлекаем «сырое» сырьё:
       MFCC[0..19], Δ-MFCC, ΔΔ-MFCC, F0 (yin), spectral_centroid,
       spectral_rolloff, spectral_bandwidth, zero_crossing_rate,
       плюс варианты CMVN (local, per-utterance, без CMVN).
  2. Из этого сырья собираем 12 разных представлений-«экспериментов».
  3. По каждому считаем:
       - mean Fisher ratio (between²/within²) — главная метрика
       - mean ГОСТ Q по парам спикеров
       - mean within-σ, mean between-σ
  4. Рисуем сравнительную картинку по всем экспериментам.
  5. Для топ-3 — отдельно PCA, гистограммы расстояний, матрица центроидов.

Запуск:
  python analyze_separability.py              # N=150 спикеров, M=8 записей
  python analyze_separability.py 200 10       # 200×10
  python analyze_separability.py 300 10 5     # 300×10, плюс топ-5 на детали

Результат:
  analysis_output/
    summary.txt
    experiments_ranking.png
    top{1..K}_fisher_<exp>.png
    top{1..K}_pca_<exp>.png
    top{1..K}_distances_<exp>.png
    top{1..K}_centroids_<exp>.png
"""

import sys
import random
import time
import warnings
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from cv_ru_loader import load_speakers_with_audio

warnings.filterwarnings("ignore")

OUT_DIR = Path("analysis_output")
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
N_FFT = int(0.025 * SAMPLE_RATE)   # 25 ms
HOP = int(0.010 * SAMPLE_RATE)     # 10 ms


# ════════════════════════════════════════════════════════════════════════
# 1. Извлечение «сырья» из одного файла
# ════════════════════════════════════════════════════════════════════════

def _cmvn_local(mfcc, window=301):
    """Текущая локальная CMVN из pipeline."""
    n_coef, n_frames = mfcc.shape
    if n_frames < window:
        window = n_frames
    out = np.zeros_like(mfcc, dtype=np.float32)
    for c in range(n_coef):
        feat = mfcc[c]
        m = np.convolve(feat, np.ones(window) / window, mode="same")
        s = np.convolve(np.abs(feat - m), np.ones(window) / window, mode="same") + 1e-8
        out[c] = (feat - m) / s
    return out


def _vad_mask(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    thr = max(0.01, np.percentile(rms, 20))
    mask = rms > thr
    if mask.sum() < 30:
        return np.ones_like(mask, dtype=bool)
    return mask


def extract_raw(path):
    """Извлечь всё нужное за один проход. Возвращает dict с признаками."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # MFCC[0..19] на ПОЛНОМ сигнале
    mfcc20 = librosa.feature.mfcc(
        y=y_pre, sr=SAMPLE_RATE, n_mfcc=20,
        n_fft=N_FFT, hop_length=HOP, n_mels=40,
        fmin=20, fmax=8000, window="hamming", center=True, norm="ortho"
    ).astype(np.float32)  # (20, T)

    # Деривативы и CMVN считаем на ПОЛНОЙ MFCC (правильный временной контекст)
    delta1_full = librosa.feature.delta(mfcc20, order=1)
    delta2_full = librosa.feature.delta(mfcc20, order=2)
    mfcc_cmvn_full = _cmvn_local(mfcc20, window=301)

    # VAD
    vad = _vad_mask(y_pre, SAMPLE_RATE)
    L = min(vad.shape[0], mfcc20.shape[1])
    vad = vad[:L]
    mfcc20 = mfcc20[:, :L]
    mfcc_cmvn_full = mfcc_cmvn_full[:, :L]
    delta1_full = delta1_full[:, :L]
    delta2_full = delta2_full[:, :L]

    # Применяем VAD ко всем матрицам синхронно
    if vad.sum() > 5:
        mfcc_cmvn = mfcc_cmvn_full[:, vad]
        mfcc_raw_active = mfcc20[:, vad]
        delta1 = delta1_full[:, vad]
        delta2 = delta2_full[:, vad]
    else:
        mfcc_cmvn = mfcc_cmvn_full
        mfcc_raw_active = mfcc20
        delta1 = delta1_full
        delta2 = delta2_full

    # Per-utterance CMVN (по активным кадрам)
    mfcc_uttcmvn = (mfcc_raw_active - mfcc_raw_active.mean(axis=1, keepdims=True)) / \
                   (mfcc_raw_active.std(axis=1, keepdims=True) + 1e-8)

    # Pitch (yin)
    try:
        f0 = librosa.yin(y_pre, fmin=70, fmax=400, sr=SAMPLE_RATE,
                         frame_length=N_FFT * 4, hop_length=HOP * 2)
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
    except Exception:
        f0 = np.array([])

    # Спектральные дескрипторы (по всему сигналу)
    sc = librosa.feature.spectral_centroid(y=y_pre, sr=SAMPLE_RATE,
                                           n_fft=N_FFT, hop_length=HOP)[0]
    sr_roll = librosa.feature.spectral_rolloff(y=y_pre, sr=SAMPLE_RATE,
                                               n_fft=N_FFT, hop_length=HOP)[0]
    sb = librosa.feature.spectral_bandwidth(y=y_pre, sr=SAMPLE_RATE,
                                            n_fft=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_pre,
                                             frame_length=N_FFT, hop_length=HOP)[0]
    def vad_apply(arr):
        L2 = min(len(arr), len(vad))
        return arr[:L2][vad[:L2]] if vad[:L2].sum() > 5 else arr[:L2]

    return {
        "mfcc_cmvn": mfcc_cmvn,
        "mfcc_uttcmvn": mfcc_uttcmvn,
        "mfcc_raw": mfcc_raw_active,
        "delta1": delta1,
        "delta2": delta2,
        "f0": f0,
        "sc": vad_apply(sc),
        "rolloff": vad_apply(sr_roll),
        "bandwidth": vad_apply(sb),
        "zcr": vad_apply(zcr),
    }


# ════════════════════════════════════════════════════════════════════════
# 2. Эксперименты — как из «сырья» собирать итоговый вектор
# ════════════════════════════════════════════════════════════════════════

def _pstat(arr, fn):
    return float(fn(arr)) if len(arr) > 0 else 0.0


EXPERIMENTS = {
    # baseline — текущий проект
    "01_baseline_mfcc13_cmvn":
        lambda f: f["mfcc_cmvn"][:13].mean(axis=1),

    # без C0 — главная гипотеза
    "02_mfcc12_no_c0_cmvn":
        lambda f: f["mfcc_cmvn"][1:13].mean(axis=1),

    # больше MFCC коэффициентов
    "03_mfcc19_no_c0_cmvn":
        lambda f: f["mfcc_cmvn"][1:20].mean(axis=1),

    # без CMVN — оценить вред локальной CMVN
    "04_mfcc12_no_c0_no_cmvn":
        lambda f: f["mfcc_raw"][1:13].mean(axis=1),

    # per-utterance CMVN
    "05_mfcc12_no_c0_utt_cmvn":
        lambda f: f["mfcc_uttcmvn"][1:13].mean(axis=1),

    # mean + std (улучшенный 26-мер)
    "06_mfcc12_mean_std_24d":
        lambda f: np.concatenate([
            f["mfcc_cmvn"][1:13].mean(axis=1),
            f["mfcc_cmvn"][1:13].std(axis=1),
        ]),

    # + дельты
    "07_mfcc12_+_delta1_24d":
        lambda f: np.concatenate([
            f["mfcc_cmvn"][1:13].mean(axis=1),
            f["delta1"][1:13].mean(axis=1),
        ]),

    # + дельты + ΔΔ
    "08_mfcc12_+_d1_d2_36d":
        lambda f: np.concatenate([
            f["mfcc_cmvn"][1:13].mean(axis=1),
            f["delta1"][1:13].mean(axis=1),
            f["delta2"][1:13].mean(axis=1),
        ]),

    # + спектральные дескрипторы
    "09_mfcc12_+_spectral_20d":
        lambda f: np.concatenate([
            f["mfcc_cmvn"][1:13].mean(axis=1),
            [_pstat(f["sc"], np.mean), _pstat(f["sc"], np.std),
             _pstat(f["rolloff"], np.mean), _pstat(f["rolloff"], np.std),
             _pstat(f["bandwidth"], np.mean), _pstat(f["bandwidth"], np.std),
             _pstat(f["zcr"], np.mean), _pstat(f["zcr"], np.std)]
        ]),

    # + pitch (F0)
    "10_mfcc12_+_pitch_15d":
        lambda f: np.concatenate([
            f["mfcc_cmvn"][1:13].mean(axis=1),
            [_pstat(f["f0"], np.mean), _pstat(f["f0"], np.std),
             _pstat(f["f0"], np.median)]
        ]),

    # КОМБО
    "11_combo_mfcc12_d1_pitch_spectral":
        lambda f: np.concatenate([
            f["mfcc_cmvn"][1:13].mean(axis=1),
            f["delta1"][1:13].mean(axis=1),
            [_pstat(f["f0"], np.mean), _pstat(f["f0"], np.std),
             _pstat(f["sc"], np.mean), _pstat(f["sc"], np.std),
             _pstat(f["zcr"], np.mean), _pstat(f["zcr"], np.std)]
        ]),

    # перцентили вместо среднего
    "12_mfcc12_percentiles_60d":
        lambda f: np.concatenate([
            np.percentile(f["mfcc_cmvn"][1:13], q, axis=1)
            for q in [10, 25, 50, 75, 90]
        ]),
}


# ════════════════════════════════════════════════════════════════════════
# 3. Метрики разделимости
# ════════════════════════════════════════════════════════════════════════

def compute_metrics(by_speaker):
    arr_all = np.vstack(list(by_speaker.values()))
    n_feat = arr_all.shape[1]

    within = np.zeros(n_feat)
    for v in by_speaker.values():
        if len(v) > 1:
            within += np.std(v, axis=0, ddof=1)
    within /= len(by_speaker)

    centroids = np.array([np.mean(v, axis=0) for v in by_speaker.values()])
    between = np.std(centroids, axis=0, ddof=1)

    fisher = (between ** 2) / (within ** 2 + 1e-12)

    # ГОСТ Q усреднённый по 300 случайным парам
    keys = list(by_speaker.keys())
    pair_q = []
    n_pairs = min(300, len(keys) * (len(keys) - 1) // 2)
    for _ in range(n_pairs):
        a, b = random.sample(keys, 2)
        own = by_speaker[a]
        E_own = np.mean(own, axis=0)
        sigma_own = np.std(own, axis=0, ddof=1) + 1e-8
        E_alien = np.mean(by_speaker[b], axis=0)
        pair_q.append(np.abs(E_alien - E_own) / sigma_own)
    mean_q = np.mean(np.array(pair_q), axis=0) if pair_q else np.zeros(n_feat)

    return {
        "n_features": n_feat,
        "n_speakers": len(by_speaker),
        "fisher": fisher,
        "mean_q_per_feature": mean_q,
        "within": within,
        "between": between,
        "mean_fisher": float(np.mean(fisher)),
        "median_fisher": float(np.median(fisher)),
        "max_fisher": float(np.max(fisher)),
        "best_features": np.argsort(fisher)[::-1][:7].tolist(),
        "mean_within": float(np.mean(within)),
        "mean_between": float(np.mean(between)),
        "mean_q_avg": float(np.mean(mean_q)),
    }


# ════════════════════════════════════════════════════════════════════════
# 4. Графики
# ════════════════════════════════════════════════════════════════════════

def plot_ranking(results):
    names = list(results.keys())
    fishers = [results[n]["mean_fisher"] for n in names]
    qs = [results[n]["mean_q_avg"] for n in names]
    dims = [results[n]["n_features"] for n in names]

    order = np.argsort(fishers)[::-1]
    names = [names[i] for i in order]
    fishers = [fishers[i] for i in order]
    qs = [qs[i] for i in order]
    dims = [dims[i] for i in order]

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    colors = ["#06D6A0" if i < 3 else "#00B4D8" for i in range(len(names))]

    axes[0].barh(range(len(names)), fishers, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([f"{n}  ({d}d)" for n, d in zip(names, dims)], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean Fisher ratio (больше — лучше)")
    axes[0].set_title("Ранжирование экспериментов по разделимости спикеров")
    axes[0].axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="Fisher=1 (граница смысла)")
    for i, v in enumerate(fishers):
        axes[0].text(v, i, f"  {v:.2f}", va="center", fontsize=9)
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(range(len(names)), qs, color=colors)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean ГОСТ Q (по парам спикеров)")
    axes[1].set_title("То же — по метрике ГОСТ Q")
    for i, v in enumerate(qs):
        axes[1].text(v, i, f"  {v:.2f}", va="center", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "experiments_ranking.png", dpi=130)
    plt.close()


def plot_feature_fisher(name, m, tag):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(m["n_features"])
    ax.bar(x, m["fisher"], color="#00B4D8")
    ax.set_title(f"{tag}: {name} — mean Fisher = {m['mean_fisher']:.2f}")
    ax.set_xlabel("Индекс признака")
    ax.set_ylabel("Fisher ratio")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    for i, v in enumerate(m["fisher"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_fisher_{name}.png", dpi=130)
    plt.close()


def plot_pca(by_speaker, name, tag, max_speakers=20):
    keys = random.sample(list(by_speaker.keys()), min(max_speakers, len(by_speaker)))
    X = np.vstack([by_speaker[k] for k in keys])
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    pca = PCA(n_components=2).fit(Xz)
    plt.figure(figsize=(9, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(keys)))
    for c, k in zip(colors, keys):
        Vz = (by_speaker[k] - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        Y = pca.transform(Vz)
        plt.scatter(Y[:, 0], Y[:, 1], color=c, alpha=0.75, s=35,
                    label=k[:8], edgecolor="black", linewidth=0.3)
    plt.legend(loc="best", fontsize=7, ncol=2)
    plt.title(f"PCA-2D — {name} (explained {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_pca_{name}.png", dpi=130)
    plt.close()


def plot_distances(by_speaker, name, tag):
    within_d, between_d = [], []
    keys = list(by_speaker.keys())
    for v in by_speaker.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                within_d.append(np.linalg.norm(v[i] - v[j]))
    n_between = min(2000, sum(len(v) for v in by_speaker.values()) * 5)
    for _ in range(n_between):
        a, b = random.sample(keys, 2)
        i = random.randrange(len(by_speaker[a]))
        j = random.randrange(len(by_speaker[b]))
        between_d.append(np.linalg.norm(by_speaker[a][i] - by_speaker[b][j]))

    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, max(max(within_d), max(between_d)), 50)
    plt.hist(within_d, bins=bins, alpha=0.65, label=f"within ({len(within_d)})",
             color="#06D6A0", density=True)
    plt.hist(between_d, bins=bins, alpha=0.65, label=f"between ({len(between_d)})",
             color="#FF6B6B", density=True)
    plt.title(f"Расстояния — {name}\n"
              f"медиана within={np.median(within_d):.3f}, between={np.median(between_d):.3f}")
    plt.xlabel("Евклидово расстояние"); plt.ylabel("Плотность"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_distances_{name}.png", dpi=130)
    plt.close()


def plot_centroids(by_speaker, name, tag, max_speakers=30):
    keys = random.sample(list(by_speaker.keys()), min(max_speakers, len(by_speaker)))
    centroids = np.array([np.mean(by_speaker[k], axis=0) for k in keys])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    cos = (centroids @ centroids.T) / (norms @ norms.T)
    plt.figure(figsize=(9, 7))
    im = plt.imshow(cos, cmap="RdYlGn_r", vmin=0, vmax=1)
    plt.colorbar(im, label="cosine similarity")
    plt.title(f"Близость центроидов — {name}\n(красное = неразличимы)")
    plt.xlabel("Спикер"); plt.ylabel("Спикер")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_centroids_{name}.png", dpi=130)
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# 5. main
# ════════════════════════════════════════════════════════════════════════

def main():
    n_speakers = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    records_per_speaker = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    random.seed(42)
    np.random.seed(42)

    print(f"[1/4] Загрузка списка спикеров…")
    speakers = load_speakers_with_audio()
    valid = [sid for sid, paths in speakers.items() if len(paths) >= records_per_speaker]
    selected = random.sample(valid, min(n_speakers, len(valid)))
    print(f"      доступно {len(valid)} спикеров (≥{records_per_speaker} записей)")
    print(f"      берём {len(selected)} × {records_per_speaker} = "
          f"{len(selected) * records_per_speaker} файлов")

    # 1. Извлекаем «сырьё» — один проход
    print(f"\n[2/4] Извлекаю признаки (один проход)…")
    raw_cache = {}  # sid → [feats per record]
    t0 = time.time()
    total = len(selected) * records_per_speaker
    done = 0
    last_print = t0
    for sid in selected:
        paths = random.sample(speakers[sid], records_per_speaker)
        feats = []
        for p in paths:
            try:
                feats.append(extract_raw(p))
            except Exception as e:
                print(f"  [warn] {sid[:8]}/{Path(p).name}: {e}")
            done += 1
            now = time.time()
            if now - last_print > 5:
                rate = done / (now - t0)
                eta = (total - done) / max(rate, 0.1)
                print(f"  {done}/{total}  ({rate:.1f}/s, ETA {eta/60:.1f} мин)")
                last_print = now
        if len(feats) >= 2:
            raw_cache[sid] = feats
    elapsed = time.time() - t0
    print(f"      готово за {elapsed/60:.1f} мин, спикеров с ≥2 записей: {len(raw_cache)}")

    # 2. Для каждого эксперимента — собираем векторы и считаем метрики
    print(f"\n[3/4] Считаю {len(EXPERIMENTS)} экспериментов…")
    results = {}
    raw_vectors = {}  # name → by_speaker

    for exp_name, exp_fn in EXPERIMENTS.items():
        by_speaker = {}
        for sid, feats_list in raw_cache.items():
            vecs = []
            for f in feats_list:
                try:
                    v = exp_fn(f)
                    v = np.asarray(v, dtype=np.float64)
                    if np.any(~np.isfinite(v)):
                        continue
                    vecs.append(v)
                except Exception as e:
                    pass
            if len(vecs) >= 2:
                by_speaker[sid] = np.array(vecs)
        if len(by_speaker) < 10:
            print(f"  [skip] {exp_name}: мало спикеров ({len(by_speaker)})")
            continue
        m = compute_metrics(by_speaker)
        results[exp_name] = m
        raw_vectors[exp_name] = by_speaker
        print(f"  {exp_name:40s}  dim={m['n_features']:3d}  "
              f"Fisher={m['mean_fisher']:.2f}  Q={m['mean_q_avg']:.2f}")

    # 3. Рисуем общий рейтинг
    print(f"\n[4/4] Строю графики…")
    plot_ranking(results)

    # 3.5. Топ-K — детали
    ranking = sorted(results.items(), key=lambda kv: -kv[1]["mean_fisher"])
    print(f"\nТоп-{top_k} по Fisher:")
    for i, (name, m) in enumerate(ranking[:top_k]):
        tag = f"top{i+1}"
        plot_feature_fisher(name, m, tag)
        plot_pca(raw_vectors[name], name, tag)
        plot_distances(raw_vectors[name], name, tag)
        plot_centroids(raw_vectors[name], name, tag)
        print(f"  {i+1}. {name}: Fisher={m['mean_fisher']:.2f}, Q={m['mean_q_avg']:.2f}")

    # 4. Текстовая сводка
    lines = []
    lines.append("=" * 80)
    lines.append("СРАВНЕНИЕ СПОСОБОВ ПРЕДОБРАБОТКИ — Common Voice RU")
    lines.append(f"  Спикеров: {len(raw_cache)} (из {n_speakers} запрошенных)")
    lines.append(f"  Записей на спикера: {records_per_speaker}")
    lines.append(f"  Всего файлов обработано: {sum(len(v) for v in raw_cache.values())}")
    lines.append(f"  Время экстракции: {elapsed/60:.1f} мин")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Эксперимент':<42} {'dim':>4} {'Fisher':>8} {'Q':>6} {'within':>8} {'between':>8}")
    lines.append("-" * 80)
    for name, m in ranking:
        lines.append(f"{name:<42} {m['n_features']:>4} "
                     f"{m['mean_fisher']:>8.3f} {m['mean_q_avg']:>6.3f} "
                     f"{m['mean_within']:>8.3f} {m['mean_between']:>8.3f}")
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"ТОП-{top_k}:")
    for i, (name, m) in enumerate(ranking[:top_k]):
        lines.append(f"\n#{i+1}: {name}  (dim={m['n_features']})")
        lines.append(f"  Mean Fisher  : {m['mean_fisher']:.3f}")
        lines.append(f"  Median Fisher: {m['median_fisher']:.3f}")
        lines.append(f"  Max Fisher   : {m['max_fisher']:.3f}")
        lines.append(f"  Mean ГОСТ Q  : {m['mean_q_avg']:.3f}")
        lines.append(f"  Топ-7 призн. : {m['best_features']}")
        lines.append(f"  Fisher по топ-7 признакам:")
        for idx in m["best_features"]:
            lines.append(f"    F{idx+1:2d}: Fisher={m['fisher'][idx]:6.2f}  "
                         f"Q={m['mean_q_per_feature'][idx]:5.2f}  "
                         f"within={m['within'][idx]:.3f}  "
                         f"between={m['between'][idx]:.3f}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("ВЫВОД:")
    best_name, best_m = ranking[0]
    baseline_m = results.get("01_baseline_mfcc13_cmvn")
    if baseline_m:
        gain = best_m["mean_fisher"] / max(baseline_m["mean_fisher"], 1e-6)
        lines.append(f"  Лучший: {best_name}")
        lines.append(f"  Выигрыш над baseline (mfcc13_cmvn): ×{gain:.2f}")
        if gain > 1.5:
            lines.append("  → СУЩЕСТВЕННЫЙ выигрыш — стоит переходить.")
        elif gain > 1.15:
            lines.append("  → УМЕРЕННЫЙ выигрыш — переход обоснован.")
        else:
            lines.append("  → Разница в пределах шума, базовый вариант ок.")
    lines.append("=" * 80)

    out_text = "\n".join(lines)
    print("\n" + out_text)
    (OUT_DIR / "summary.txt").write_text(out_text, encoding="utf-8")
    print(f"\n✅ Готово. Все артефакты в: {OUT_DIR}/")


if __name__ == "__main__":
    main()