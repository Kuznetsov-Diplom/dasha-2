#!/usr/bin/env python3
"""
Dasha v2.5 — Нейросетевой преобразователь биометрия-код по ГОСТ Р 52633.5-2011

Вкладки:
  1. Обработка фразы     — сигнал, MFCC, нормализованный вектор
  2. Корреляция          — матрица корреляций, эталон спикера
  3. Нормализатор (ГОСТ) — обучение global_minmax_abs на датасете
  4. НПБК — Регистрация  — обучение, debug ГОСТ-формул, метрики
  5. НПБК — Восстановление — голос → ключ → расшифровка

Исправления v2.5:
  - protected_secret НЕ выводится в открытом виде
  - reg_metrics разбиты на 3 отдельных компонента
  - Нормализатор обучается явно (вкладка 3) перед НПБК
  - Race-condition в trained_speakers устранён через lock
  - Улучшена обработка ошибок (нет голого except: continue)
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import random
import json
import secrets
import threading
from pathlib import Path
import soundfile as sf
import psycopg2

try:
    from pipeline import VoiceFeaturePipeline
    from normalizer import FeatureNormalizer
    from cv_ru_loader import load_speakers_with_audio, CVRuLoader
    from npbk import NPBK
except ImportError as e:
    print(f"Import error: {e}")
    raise

# ── Глобальные объекты ──────────────────────────────────────────────────────
pipeline = VoiceFeaturePipeline(use_rasta=False, use_deltas=False,
                                use_cmvn=False, drop_c0=True)
npbk = NPBK(key_bits=128)
loader = CVRuLoader()

if not pipeline.normalizer.is_fitted():
    print("[App] ⚠️ Нормализатор не обучен (models/audio_params/normalizer_params.json "
          "отсутствует). Обучите его на вкладке 3 — без этого pipeline использует "
          "локальный fallback и качество НПБК будет ниже.")

global_speakers = load_speakers_with_audio()
speaker_list = sorted([sid for sid, paths in global_speakers.items() if len(paths) >= 5])

_trained_lock = threading.Lock()
trained_speakers: set = set()
TRAINED_FILE = Path("models/trained_speakers.json")
if TRAINED_FILE.exists():
    trained_speakers = set(json.load(open(TRAINED_FILE)))


def save_trained():
    TRAINED_FILE.parent.mkdir(exist_ok=True)
    with _trained_lock:
        json.dump(list(trained_speakers), open(TRAINED_FILE, "w"))


def get_available_speakers():
    with _trained_lock:
        return [s for s in speaker_list if s not in trained_speakers]


def get_random_available_speaker():
    avail = get_available_speakers()
    return random.choice(avail) if avail else None


def get_nbk_records():
    """Список НБК для дропдауна на вкладке восстановления.
    Возвращает строки вида 'user_id | source_type | created_at'.
    speaker_id берётся отдельно через get_nbk_speaker_id().
    """
    try:
        conn = psycopg2.connect(npbk.db_url)
        cur = conn.cursor()
        cur.execute("ALTER TABLE npbk_containers ADD COLUMN IF NOT EXISTS source_speaker_id TEXT")
        conn.commit()
        cur.execute("""
            SELECT user_id, source_type, source_speaker_id, created_at
            FROM npbk_containers ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        labels = []
        for user_id, src_type, src_speaker, created in rows:
            spk_tag = f"speaker={src_speaker[:12]}…" if src_speaker else "voice=upload"
            labels.append(f"{user_id} | {src_type} | {spk_tag} | {created}")
        return labels
    except Exception as e:
        print(f"[DB] get_nbk_records error: {e}")
        return []


def get_nbk_speaker_id(user_id: str):
    """Вернуть source_speaker_id для данного user_id или None."""
    try:
        conn = psycopg2.connect(npbk.db_url)
        cur = conn.cursor()
        cur.execute(
            "SELECT source_speaker_id FROM npbk_containers WHERE user_id=%s",
            (user_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"[DB] get_nbk_speaker_id error: {e}")
        return None


def generate_secret():
    return "psk_" + secrets.token_urlsafe(12)


# ── Вспомогательные графики ─────────────────────────────────────────────────

def plot_signal(y, sr, title="Предобработанный сигнал"):
    t = np.linspace(0, len(y) / sr, len(y))
    fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", line=dict(color="#00B4D8", width=0.8)))
    fig.update_layout(
        title=title,
        xaxis_title="Время (с)", yaxis_title="Амплитуда",
        height=220, template="plotly_dark", margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig


def plot_mfcc(mfcc, title="MFCC (13 коэф.)"):
    fig = px.imshow(
        mfcc, aspect="auto", color_continuous_scale="Viridis",
        labels=dict(x="Кадры", y="MFCC коэф. (1-13)", color="Значение"),
        title=title
    )
    fig.update_layout(height=250, template="plotly_dark", margin=dict(l=50, r=20, t=40, b=40))
    return fig


def plot_vector(vector, title="Нормализованный вектор [0,1]", color="#FF6B6B"):
    labels = [f"F{i+1}" for i in range(len(vector))]
    fig = go.Figure(go.Bar(
        x=labels, y=vector,
        marker_color=color,
        text=[f"{v:.3f}" for v in vector],
        textposition="outside"
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1.15]),
        height=300, template="plotly_dark",
        margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig


def plot_correlation_matrix(vectors, title="Матрица корреляций векторов (0–1)"):
    arr = np.array(vectors)
    n = len(arr)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ni = np.linalg.norm(arr[i])
            nj = np.linalg.norm(arr[j])
            corr[i, j] = np.dot(arr[i], arr[j]) / (ni * nj + 1e-8)
    labels = [f"Запись {i+1}" for i in range(n)]
    fig = px.imshow(
        corr, x=labels, y=labels,
        color_continuous_scale="RdYlGn", zmin=0, zmax=1,
        title=title
    )
    fig.update_layout(height=380, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_vectors_vs_mean(vectors, title="Векторы vs Средний эталон"):
    arr = np.array(vectors)
    mean_v = np.mean(arr, axis=0)
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, v in enumerate(arr):
        fig.add_trace(go.Scatter(
            y=v, mode="lines+markers",
            name=f"Запись {i+1}",
            line=dict(color=colors[i % len(colors)], width=1),
            marker=dict(size=4)
        ))
    fig.add_trace(go.Scatter(
        y=mean_v, mode="lines",
        name="Эталон (среднее)",
        line=dict(color="white", width=2.5, dash="dash")
    ))
    fig.update_layout(
        title=title, height=320, template="plotly_dark",
        margin=dict(l=50, r=10, t=40, b=40)
    )
    return fig


def plot_normalizer_comparison(raw_vectors, norm_vectors, feature_idx=0):
    """График: распределение одного признака до и после нормализации"""
    raw_vals = [v[feature_idx] for v in raw_vectors]
    norm_vals = [v[feature_idx] for v in norm_vectors]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=raw_vals, name="До нормализации", opacity=0.7,
                               marker_color="#FF6B6B", nbinsx=30))
    fig.add_trace(go.Histogram(x=norm_vals, name="После нормализации", opacity=0.7,
                               marker_color="#00B4D8", nbinsx=30))
    fig.update_layout(
        title=f"Признак F{feature_idx+1}: распределение до/после global_minmax_abs",
        barmode="overlay", height=280, template="plotly_dark",
        margin=dict(l=50, r=20, t=50, b=40)
    )
    return fig


def plot_key_bits(binary_str, title="Internal key (биты НПБК)"):
    bits = [int(b) for b in binary_str[:64]]
    fig = go.Figure(go.Bar(
        x=list(range(len(bits))), y=bits,
        marker_color=["#00B4D8" if b else "#FF6B6B" for b in bits]
    ))
    fig.update_layout(
        title=title, yaxis=dict(range=[0, 1.2]),
        height=180, template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=30)
    )
    return fig


# ── Обработка фразы (вкладка 1) ─────────────────────────────────────────────

def process_phrase(audio_input, speaker_id, mode, progress=gr.Progress()):
    """Извлечь вектор из аудио и показать всю цепочку"""
    progress(0, desc="Загрузка аудио...")
    path = None

    if mode == "Своя запись" and audio_input is not None:
        sr, y = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            path = tmp.name
    elif mode == "Из датасета" and speaker_id:
        paths = global_speakers.get(speaker_id, [])
        if not paths:
            return "Спикер не найден", None, None, None
        path = random.choice(paths)
    else:
        return "Выберите источник аудио", None, None, None

    progress(0.3, desc="Извлечение признаков...")
    try:
        res = pipeline.extract_features(path)
    except Exception as e:
        return f"Ошибка обработки: {e}", None, None, None

    progress(0.7, desc="Построение графиков...")
    y_pre = res["y_pre"]
    sr_val = res["sr"]
    mfcc = res["mfcc_norm"]
    vec = res["normalized_vector"]
    raw_vec = res["raw_mean_vector"]
    vad_pct = float(np.mean(res["vad_mask"])) * 100

    sig_plot = plot_signal(y_pre, sr_val)
    mfcc_plot = plot_mfcc(mfcc)
    vec_plot = plot_vector(vec, title=f"Нормализованный 13-мерный вектор [0,1] (VAD: {vad_pct:.0f}%)")

    info = f"""
**✅ Вектор извлечён**

| Параметр | Значение |
|---|---|
| Размерность | {res['dim']} ({res['dim_label']}) |
| VAD: речь | {vad_pct:.1f}% кадров |
| Длина записи | {len(y_pre)/sr_val:.2f} с |
| Pipeline | {res['pipeline_version']} |

**Raw вектор (до нормализации):**
`{[round(x, 4) for x in raw_vec]}`

**Нормализованный вектор:**
`{[round(x, 4) for x in vec]}`
"""
    progress(1.0)
    return info, sig_plot, mfcc_plot, vec_plot


# ── Корреляция и стабильность (вкладка 2) ───────────────────────────────────

def build_correlation(mode, speaker_id, audio_files, progress=gr.Progress()):
    progress(0, desc="Загрузка записей...")
    vectors = []
    paths = []

    if mode == "Из датасета" and speaker_id:
        all_paths = global_speakers.get(speaker_id, [])
        paths = random.sample(all_paths, min(10, len(all_paths)))
    elif audio_files:
        paths = audio_files if isinstance(audio_files, list) else [audio_files]
    else:
        return "Выберите источник", None, None, ""

    for i, p in enumerate(paths):
        progress(i / len(paths) * 0.7, desc=f"Обработка {i+1}/{len(paths)}...")
        try:
            res = pipeline.extract_features(p)
            vectors.append(res["normalized_vector"])
        except Exception as e:
            print(f"[Corr] Ошибка файла {p}: {e}")

    if len(vectors) < 2:
        return f"Недостаточно записей ({len(vectors)}). Нужно минимум 2.", None, None, ""

    progress(0.8, desc="Расчёт корреляций...")
    arr = np.array(vectors)
    corr_plot = plot_correlation_matrix(vectors)
    vec_plot = plot_vectors_vs_mean(vectors)

    # Средняя межзаписевая корреляция
    sims = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            ni, nj = np.linalg.norm(arr[i]), np.linalg.norm(arr[j])
            sims.append(np.dot(arr[i], arr[j]) / (ni * nj + 1e-8))
    mean_corr = np.mean(sims)
    std_corr = np.std(sims)

    # Стабильность по каждому признаку
    feat_std = np.std(arr, axis=0)
    stable_features = int(np.sum(feat_std < 0.05))

    summary = f"""
**📊 Анализ стабильности ({len(vectors)} записей)**

| Метрика | Значение |
|---|---|
| Средняя корреляция | **{mean_corr:.3f}** |
| Разброс корреляций | ±{std_corr:.3f} |
| Стабильных признаков (σ<0.05) | {stable_features} / 13 |
| Оценка пригодности для НПБК | {'✅ Хорошо' if mean_corr > 0.85 else '⚠️ Удовлетворительно' if mean_corr > 0.7 else '❌ Плохо'} |
"""
    progress(1.0)
    return summary, corr_plot, vec_plot, f"Средняя корреляция: **{mean_corr:.3f}**"


# ── Нормализатор ГОСТ (вкладка 3) ───────────────────────────────────────────

def train_normalizer(num_speakers, progress=gr.Progress()):
    """Обучить global_minmax_abs нормализатор на реальных данных датасета"""
    progress(0, desc="Подготовка данных...")
    normalizer = FeatureNormalizer(method="global_minmax_abs")

    valid = [sid for sid, p in global_speakers.items() if len(p) >= 3]
    selected = random.sample(valid, min(int(num_speakers), len(valid)))

    all_raw = []
    errors = 0
    for i, sid in enumerate(selected):
        progress(i / len(selected) * 0.7, desc=f"Спикер {i+1}/{len(selected)}...")
        for p in random.sample(global_speakers[sid], min(5, len(global_speakers[sid]))):
            try:
                res = pipeline.extract_features(p)
                all_raw.append(res["features"])
            except Exception as e:
                errors += 1

    if len(all_raw) < 50:
        return f"Мало данных ({len(all_raw)} векторов). Попробуйте увеличить число спикеров.", None, None, ""

    progress(0.75, desc="Обучение нормализатора...")
    arr = np.array(all_raw)
    normalizer.fit(arr, pipeline_tag=getattr(pipeline, "PIPELINE_TAG", None))

    progress(0.85, desc="Сохранение параметров...")
    normalizer.save()
    # Перезагружаем в pipeline
    pipeline.normalizer = normalizer

    # Нормализуем для визуализации
    norm_vecs = [normalizer.transform(v).tolist() for v in all_raw[:200]]
    raw_sample = [v.tolist() for v in all_raw[:200]]

    progress(0.95, desc="Построение графиков...")
    hist_plot = plot_normalizer_comparison(raw_sample, norm_vecs, feature_idx=0)

    # Coverage: сколько признаков в [0,1]
    norm_arr = np.array(norm_vecs)
    in_range = np.sum((norm_arr >= 0) & (norm_arr <= 1)) / norm_arr.size * 100

    params = normalizer.params
    minv = params["min"]
    maxv = params["max"]
    ranges = [round(maxv[i] - minv[i], 4) for i in range(13)]

    range_plot = go.Figure(go.Bar(
        x=[f"F{i+1}" for i in range(13)],
        y=ranges,
        marker_color="#00B4D8",
        text=[f"{r:.3f}" for r in ranges],
        textposition="outside"
    ))
    range_plot.update_layout(
        title="Диапазон каждого признака (max - min) по датасету",
        height=280, template="plotly_dark",
        margin=dict(l=50, r=20, t=50, b=40)
    )

    summary = f"""
**✅ Нормализатор обучен и сохранён**

| Параметр | Значение |
|---|---|
| Метод | `global_minmax_abs` (ГОСТ Р 52633.5) |
| Спикеров использовано | {len(selected)} |
| Векторов для обучения | {len(all_raw)} |
| Ошибок при обработке | {errors} |
| Покрытие [0,1] после нормализации | **{in_range:.1f}%** |
| Файл сохранён | `models/audio_params/normalizer_params.json` |

**Диапазоны признаков (raw):**
```
min: {[round(x, 3) for x in minv]}
max: {[round(x, 3) for x in maxv]}
```

> Нормализатор автоматически загрузится в pipeline для вкладки НПБК.
"""
    progress(1.0)
    return summary, hist_plot, range_plot, "✅ Нормализатор готов"


# ── НПБК — Регистрация (вкладка 4) ──────────────────────────────────────────

def morph_augment(vectors, target_count=11):
    """
    Линейный морфинг по ГОСТ Р 52633.2 — равномерные t_k = k/(N+1)
    для каждой пары родителей.
    """
    if len(vectors) >= target_count:
        return vectors
    augmented = [np.array(v, dtype=np.float64) for v in vectors]
    src = list(augmented)
    n_src = len(src)

    if n_src < 2:
        base = src[0] if src else np.zeros(13)
        while len(augmented) < target_count:
            augmented.append((base + np.random.normal(0, 0.02, len(base))))
        return [v.tolist() for v in augmented[:target_count]]

    # Все пары + равномерные t_k. n_children подбирается так, чтобы хватило.
    pairs = [(i, j) for i in range(n_src) for j in range(i + 1, n_src)]
    needed = target_count - len(augmented)
    per_pair = max(1, -(-needed // len(pairs)))  # ceil
    for (i, j) in pairs:
        A, B = src[i], src[j]
        for k in range(1, per_pair + 1):
            t = k / (per_pair + 1)
            augmented.append(A + t * (B - A))
            if len(augmented) >= target_count:
                break
        if len(augmented) >= target_count:
            break

    return [v.tolist() for v in augmented[:target_count]]


def register_npbk(mode, audio_files, speaker_id, user_name, desired_key, progress=gr.Progress()):
    progress(0, desc="Подготовка...")

    if not user_name:
        user_name = speaker_id or ("user_" + str(random.randint(1000, 9999)))

    if not desired_key:
        return (
            "❌ Укажите ваш секрет (protected_secret)",
            None, None,
            "—", "—", "—",
            "", ""
        )

    vectors = []
    paths = []

    if mode == "Из датасета" and speaker_id:
        with _trained_lock:
            if speaker_id in trained_speakers:
                return (
                    "❌ Спикер уже обучен. Выберите другого.",
                    None, None, "—", "—", "—", "", ""
                )
        all_paths = global_speakers.get(speaker_id, [])
        paths = random.sample(all_paths, min(12, len(all_paths)))
    elif audio_files:
        paths = audio_files if isinstance(audio_files, list) else [audio_files]
    else:
        return "❌ Выберите источник данных", None, None, "—", "—", "—", "", ""

    progress(0.15, desc="Извлечение векторов...")
    for p in paths:
        try:
            res = pipeline.extract_features(p)
            vectors.append(res["normalized_vector"])
        except Exception as e:
            print(f"[Register] Ошибка файла {p}: {e}")

    if len(vectors) < 8:
        vectors = morph_augment(vectors, target_count=11)

    if len(vectors) < 8:
        return (
            f"❌ Мало записей даже после морфинга ({len(vectors)}). Нужно минимум 8.",
            None, None, "—", "—", "—", "", ""
        )

    if len(vectors) < 11:
        old_len = len(vectors)
        vectors = morph_augment(vectors, target_count=11)
        progress(0.25, desc=f"Размножено до 11 (морфинг по ГОСТ): {old_len} → {len(vectors)}")

    progress(0.4, desc="Сбор базы «Чужой»...")
    alien = []
    other_speakers = [s for s in speaker_list if s != speaker_id][:6]
    for osid in other_speakers:
        osid_paths = global_speakers.get(osid, [])
        for p in random.sample(osid_paths, min(12, len(osid_paths))):
            try:
                v = pipeline.extract_features(p)["normalized_vector"]
                alien.append(v)
            except Exception as e:
                print(f"[Register] Чужой {osid}: {e}")

    # ГОСТ требует минимум 64 реальных биометрических образа «Чужой»
    if len(alien) < 64:
        # Добираем морфингом существующих чужих (не рандомом!)
        if len(alien) >= 2:
            alien = morph_augment(alien, target_count=64)
        else:
            return (
                f"❌ Недостаточно данных «Чужой» ({len(alien)}). Нужно минимум 64.",
                None, None, "—", "—", "—", "", ""
            )

    progress(0.55, desc="Обучение НПБК (формулы ГОСТ)...")
    src_speaker = speaker_id if (mode == "Из датасета" and speaker_id) else None
    try:
        success, quality = npbk.train(
            vectors, alien,
            user_id=user_name,
            protected_secret=desired_key,
            source_speaker_id=src_speaker,
            debug=True
        )
    except Exception as e:
        return f"❌ Ошибка обучения: {e}", None, None, "—", "—", "—", "", ""

    debug = getattr(npbk, "debug_info", {}) or {}

    # ── Формируем DEBUG блок (без секрета!) ──────────────────────────────────
    debug_md = ""
    if debug:
        debug_md = f"""
---
**📊 DEBUG — ГОСТ Р 52633.5, формулы (4)(6)(7)**

```
Формула (4): Q(V_i) = |E_чужой - E_свой| / σ_свой
Формула (6): μ_i    = Q(V_i) / σ_Чужой(V_i)
Формула (7): sign(μ_i) = sign(E_свой - E_чужой)
Bias (μ₀):  = -E_чужой(Σ μ_i · v_i)
```

| Признак | E_свой | σ_свой | E_чужой | σ_чужой | Q(V_i) |
|---|---|---|---|---|---|
{chr(10).join(f"| F{i+1} | {debug['E_own'][i]:.4f} | {debug['sigma_own'][i]:.4f} | {debug['E_alien'][i]:.4f} | {debug['sigma_alien'][i]:.4f} | {debug['q_per_feature'][i]:.4f} |" for i in range(13))}

**Топ-7 признаков:** {debug.get('top7_feature_indices', [])}

**Bias:** mean={debug.get('bias_mean', 0):.4f}, std={debug.get('bias_std', 0):.4f}

**3 примера «Свой»:**
```
{debug.get('sample_own_vectors', [])}
```
**3 примера «Чужой»:**
```
{debug.get('sample_alien_vectors', [])}
```
"""

    progress(0.8, desc="Расчёт метрик...")

    if not success:
        with _trained_lock:
            trained_speakers.discard(speaker_id or user_name)
        db_err = quality.get("db_error")
        if db_err:
            fail_md = f"""
**❌ Обучение прошло, но не удалось сохранить в БД**

| Метрика | Значение |
|---|---|
| FRR | {quality['FRR']:.1%} |
| FAR | {quality['FAR']:.1%} |

**Причина:** `{db_err}`

Проверьте, что PostgreSQL запущен:
```
docker compose up -d db
```
{debug_md}
"""
        else:
            fail_md = f"""
**❌ Обучение провалено**

| Метрика | Значение | Норма |
|---|---|---|
| FRR | {quality['FRR']:.1%} | < 12% |
| FAR | {quality['FAR']:.1%} | < 8% |
| mean\\|μ\\| | {quality.get('mean_|mu|', 0):.3f} | < 30 |
| mean_Q | {quality.get('mean_Q', 0):.4f} | — |

**Рекомендации:**
- Малое sigma_own → записи слишком однородны (попробуйте другого спикера)
- Низкий Q → признаки плохо различают Свой/Чужой
{debug_md}
"""
        return fail_md, None, None, "—", "—", "—", "", ""

    with _trained_lock:
        trained_speakers.add(speaker_id or user_name)
    save_trained()

    progress(0.9, desc="Визуализация...")
    vec_plot = plot_vector(vectors[0], title="Входной вектор (первая запись)")
    internal_key = npbk.generate_key(vectors[0])
    key_plot = plot_key_bits(internal_key)

    foreign_info = ", ".join([
        f"{s[:12]}… ({len(global_speakers.get(s, []))} фраз)"
        for s in other_speakers[:4]
    ])

    # Метрики для трёх отдельных компонентов
    layer1_q = round(np.mean([abs(np.mean(v) - 0.5) for v in vectors]), 4)
    layer2_q = round(float(quality.get("mean_Q", 0)), 4)

    # Реальный EER: скоры близости к target_key через долю совпадающих бит.
    target_bits = (debug or {}).get("target_key", "")
    if target_bits and npbk.trained:
        def _bit_score(v):
            k = npbk.generate_key(v)
            n = min(len(k), len(target_bits))
            return sum(1 for i in range(n) if k[i] == target_bits[i]) / n if n else 0.0
        own_scores = [_bit_score(v) for v in vectors]
        alien_scores = [_bit_score(v) for v in alien[:80]]
        try:
            eer_val = float(pipeline.compute_eer(own_scores, alien_scores))
        except Exception:
            eer_val = float("nan")
    else:
        eer_val = float("nan")

    success_md = f"""
**✅ Обучение успешно! (Dasha v2.5)**

| Метрика | Значение |
|---|---|
| Пользователь | **{user_name}** |
| FRR | **{quality['FRR']:.1%}** |
| FAR | **{quality['FAR']:.1%}** |
| EER | {eer_val} |
| mean\\|μ\\| | {quality.get('mean_|mu|', 0):.3f} |
| mean_Q | {quality.get('mean_Q', 0):.4f} |
| База «Чужой» | {foreign_info} ({len(alien)} примеров) |
| Версия | {quality.get('version', '—')} |

> 🔒 Ваш секрет зашифрован и сохранён в PostgreSQL.
> Чтобы получить его обратно — перейдите на вкладку **Восстановление**.

{debug_md}
"""
    progress(1.0)
    return (
        success_md,
        vec_plot,
        key_plot,
        f"Слой 1 (качество): {layer1_q}",
        f"Слой 2 (mean_Q): {layer2_q}",
        f"EER: {eer_val}",
        internal_key,
        "✅ Ключ защищён в PostgreSQL"
    )


# ── НПБК — Восстановление (вкладка 5) ───────────────────────────────────────

def recover_key(nbk_record, mode, audio, dataset_audio_path, progress=gr.Progress()):
    progress(0, desc="Загрузка НПБК...")
    if not nbk_record:
        return "Выберите запись из НБК", None, "—", ""

    user_id = nbk_record.split(" | ")[0]
    if not npbk.load_from_db(user_id):
        return f"Не удалось загрузить НПБК для {user_id}", None, "—", ""

    progress(0.3, desc="Обработка голоса...")

    # Источник аудио
    path = None
    if mode == "Из датасета (тот же спикер)":
        if not dataset_audio_path:
            return ("Выберите запись из датасета (или у НБК нет привязанного "
                    "спикера — переключите режим на «Микрофон/файл»)"), None, "—", ""
        path = dataset_audio_path
    else:
        if audio is None:
            return "Загрузите запись голоса", None, "—", ""
        sr, y = audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            path = tmp.name

    try:
        res = pipeline.extract_features(path)
        vec = res["normalized_vector"]
    except Exception as e:
        return f"Ошибка обработки голоса: {e}", None, "—", ""

    progress(0.6, desc="Восстановление через НПБК...")
    try:
        internal_key = npbk.generate_key(vec)
        if npbk.encrypted_secret:
            original_secret = npbk.decrypt_secret(internal_key)
        else:
            original_secret = "(старый формат — секрет не сохранён)"
    except Exception as e:
        return f"Ошибка восстановления: {e}", None, "—", ""

    progress(1.0)
    vec_plot = plot_vector(vec, title="Входной вектор при восстановлении", color="#00B4D8")

    rec_md = f"""
**✅ Восстановление успешно!**

| Параметр | Значение |
|---|---|
| Пользователь | **{user_id}** |
| Привязанный спикер | `{npbk.source_speaker_id or '—'}` |
| Источник звука | {mode} |
| Internal key (preview) | `{internal_key[:32]}…` |

> Ваш оригинальный секрет отображён ниже в защищённом поле.
"""
    return rec_md, vec_plot, original_secret, "✅ Ключ восстановлен"


def get_dataset_phrases_for_nbk(nbk_record):
    """По выбранной записи НБК — список (label, path) фраз привязанного спикера."""
    if not nbk_record:
        return gr.update(choices=[], value=None, visible=False), ""
    user_id = nbk_record.split(" | ")[0]
    speaker = get_nbk_speaker_id(user_id)
    if not speaker:
        return (
            gr.update(choices=[], value=None, visible=False),
            "_У этой записи нет привязанного спикера датасета (была регистрация загруженными файлами)._"
        )
    paths = global_speakers.get(speaker, [])
    if not paths:
        return (
            gr.update(choices=[], value=None, visible=True),
            f"_Спикер `{speaker[:24]}…` есть в БД, но его аудио нет в датасете локально._"
        )
    # Метка = просто имя файла, value = полный путь
    choices = [(Path(p).name, p) for p in paths[:20]]
    return (
        gr.update(choices=choices, value=choices[0][1], visible=True),
        f"_Привязанный спикер: `{speaker[:24]}…` ({len(paths)} фраз доступно)._"
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
.tab-nav button { font-size: 14px; font-weight: 600; }
.metric-box { background: #1a1a2e; border-radius: 8px; padding: 12px; }
.secret-field input { font-family: monospace; }
"""

with gr.Blocks(
    title="Dasha v2.5 — НПБК по ГОСТ Р 52633.5-2011",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS
) as demo:

    gr.Markdown("""
# 🛡️ Dasha v2.5 — Нейросетевой преобразователь «биометрия → код» (ГОСТ Р 52633.5-2011)
**Цепочка:** Голос → MFCC → global_minmax_abs → НПБК → Зашифрованный секрет (Кузнечик)
""")

    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("1️⃣ Обработка фразы"):
        gr.Markdown("### Извлечение 13-мерного вектора из голосовой записи")
        with gr.Row():
            with gr.Column(scale=1):
                tab1_mode = gr.Radio(
                    ["Своя запись", "Из датасета"], value="Своя запись",
                    label="Источник"
                )
                tab1_audio = gr.Audio(
                    sources=["microphone", "upload"], type="numpy",
                    label="🎤 Запись или файл"
                )
                tab1_speaker = gr.Dropdown(
                    choices=speaker_list, label="Спикер из датасета",
                    visible=False
                )
                tab1_btn = gr.Button("🚀 Извлечь вектор", variant="primary")

            with gr.Column(scale=2):
                tab1_info = gr.Markdown()
                tab1_sig = gr.Plot(label="Сигнал")
                tab1_mfcc = gr.Plot(label="MFCC")
                tab1_vec = gr.Plot(label="Нормализованный вектор")

        def toggle_tab1(m):
            return gr.update(visible=(m == "Из датасета"))
        tab1_mode.change(toggle_tab1, inputs=[tab1_mode], outputs=[tab1_speaker])
        tab1_btn.click(
            process_phrase,
            inputs=[tab1_audio, tab1_speaker, tab1_mode],
            outputs=[tab1_info, tab1_sig, tab1_mfcc, tab1_vec]
        )

    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("2️⃣ Корреляция и стабильность"):
        gr.Markdown("### Анализ стабильности биометрического образа (несколько записей одного спикера)")
        with gr.Row():
            with gr.Column(scale=1):
                tab2_mode = gr.Radio(
                    ["Из датасета", "Загрузить файлы"], value="Из датасета",
                    label="Источник"
                )
                with gr.Group(visible=True) as tab2_ds_group:
                    tab2_speaker = gr.Dropdown(choices=speaker_list, label="Спикер")
                    tab2_rand = gr.Button("🎲 Случайный спикер", size="sm")
                with gr.Group(visible=False) as tab2_up_group:
                    tab2_files = gr.File(
                        file_count="multiple", file_types=[".wav", ".mp3"],
                        label="Файлы (8–12 записей)"
                    )
                tab2_btn = gr.Button("📊 Построить корреляцию", variant="primary")
                tab2_summary_label = gr.Markdown()

            with gr.Column(scale=2):
                tab2_summary = gr.Markdown()
                tab2_corr = gr.Plot(label="Матрица корреляций")
                tab2_vecs = gr.Plot(label="Векторы vs Эталон")

        def toggle_tab2(m):
            return gr.update(visible=(m == "Из датасета")), gr.update(visible=(m != "Из датасета"))
        tab2_mode.change(toggle_tab2, inputs=[tab2_mode], outputs=[tab2_ds_group, tab2_up_group])
        tab2_rand.click(get_random_available_speaker, outputs=[tab2_speaker])
        tab2_btn.click(
            build_correlation,
            inputs=[tab2_mode, tab2_speaker, tab2_files],
            outputs=[tab2_summary, tab2_corr, tab2_vecs, tab2_summary_label]
        )

    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("3️⃣ Нормализатор (ГОСТ)"):
        gr.Markdown("""
### Обучение нормализатора `global_minmax_abs` по ГОСТ Р 52633.5
Нормализатор приводит каждый из 13 признаков MFCC к диапазону [0, 1] на основе
реальных данных датасета. Параметры сохраняются и автоматически загружаются в pipeline.
""")
        with gr.Row():
            with gr.Column(scale=1):
                tab3_nspk = gr.Slider(
                    minimum=50, maximum=500, value=200, step=50,
                    label="Число спикеров для обучения нормализатора"
                )
                tab3_btn = gr.Button("⚙️ Обучить нормализатор", variant="primary")
                tab3_status = gr.Markdown()
                gr.Markdown("""
**Метод `global_minmax_abs`:**
```
v_norm = (v - min_global) / (max_global - min_global + ε)
v_norm = clip(v_norm, 0, 1)
```
Параметры `min` и `max` вычисляются по всей обучающей выборке.
""")
            with gr.Column(scale=2):
                tab3_info = gr.Markdown()
                tab3_hist = gr.Plot(label="Распределение F1 до/после")
                tab3_range = gr.Plot(label="Диапазоны признаков")

        tab3_btn.click(
            train_normalizer,
            inputs=[tab3_nspk],
            outputs=[tab3_info, tab3_hist, tab3_range, tab3_status]
        )

    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("4️⃣ НПБК — Регистрация"):
        gr.Markdown("""
### Регистрация нейросетевого преобразователя биометрия→код
Обучение по **ГОСТ Р 52633.5**: формулы (4)(6)(7) — качество признаков, веса, знаки.
""")
        with gr.Row():
            with gr.Column(scale=1):
                tab4_mode = gr.Radio(
                    ["Из датасета", "Загрузить файлы"], value="Из датасета",
                    label="Источник голоса"
                )
                with gr.Group(visible=True) as tab4_ds_group:
                    tab4_speaker = gr.Dropdown(
                        choices=get_available_speakers(),
                        label="Спикер из датасета (Common Voice RU)",
                        info="Только необученные"
                    )
                    tab4_rand = gr.Button("🎲 Случайный спикер", size="sm")
                with gr.Group(visible=False) as tab4_up_group:
                    tab4_files = gr.File(
                        file_count="multiple",
                        file_types=[".wav", ".mp3"],
                        label="Файлы голоса (8–12 записей)"
                    )
                tab4_user = gr.Textbox(label="Имя пользователя (user_id)", placeholder="ivan_2026")
                with gr.Row():
                    tab4_secret = gr.Textbox(
                        label="Ваш секрет (protected_secret)",
                        placeholder="Мой_ключ_ЭЦП_2026",
                        type="password",
                        elem_classes=["secret-field"]
                    )
                    tab4_gen = gr.Button("🔄", size="sm", scale=0)
                tab4_btn = gr.Button("🚀 Обучить НПБК", variant="primary", size="lg")

            with gr.Column(scale=2):
                tab4_md = gr.Markdown()
                tab4_vec = gr.Plot(label="Входной вектор")
                tab4_key = gr.Plot(label="Internal key (биты)")
                with gr.Row():
                    tab4_m1 = gr.Textbox(label="Метрика Слой 1", interactive=False)
                    tab4_m2 = gr.Textbox(label="Метрика Слой 2", interactive=False)
                    tab4_m3 = gr.Textbox(label="EER", interactive=False)
                tab4_internal = gr.Textbox(
                    label="Internal key (удаляется после сохранения)",
                    interactive=False
                )
                tab4_status = gr.Markdown()

        def toggle_tab4(m):
            return gr.update(visible=(m == "Из датасета")), gr.update(visible=(m != "Из датасета"))
        tab4_mode.change(toggle_tab4, inputs=[tab4_mode], outputs=[tab4_ds_group, tab4_up_group])
        tab4_rand.click(get_random_available_speaker, outputs=[tab4_speaker])
        tab4_speaker.change(lambda s: s or "", inputs=[tab4_speaker], outputs=[tab4_user])
        tab4_gen.click(generate_secret, outputs=[tab4_secret])

        tab4_btn.click(
            register_npbk,
            inputs=[tab4_mode, tab4_files, tab4_speaker, tab4_user, tab4_secret],
            outputs=[tab4_md, tab4_vec, tab4_key, tab4_m1, tab4_m2, tab4_m3, tab4_internal, tab4_status]
        )

    # ════════════════════════════════════════════════════════════════════════
    with gr.Tab("5️⃣ НПБК — Восстановление"):
        gr.Markdown("""
### Восстановление защищённого секрета через биометрию
Предъявите голос — НПБК восстановит ключ и расшифрует ваш секрет (Кузнечик).
Если НБК обучен на спикере из датасета — можно сразу проверить фразой
из того же спикера (без записи микрофона).
""")
        with gr.Row():
            with gr.Column(scale=1):
                tab5_refresh = gr.Button("🔄 Обновить список НБК", size="sm")
                tab5_nbk = gr.Dropdown(
                    choices=get_nbk_records(),
                    label="Обученные записи НПБК"
                )
                tab5_speaker_info = gr.Markdown()
                tab5_mode = gr.Radio(
                    ["Микрофон/файл", "Из датасета (тот же спикер)"],
                    value="Микрофон/файл",
                    label="Источник голоса для проверки"
                )
                tab5_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="🎤 Ваш голос",
                    visible=True
                )
                tab5_dataset_phrase = gr.Dropdown(
                    choices=[], label="Фраза из датасета (привязанного спикера)",
                    visible=False
                )
                tab5_btn = gr.Button("🔑 Восстановить ключ", variant="primary", size="lg")

            with gr.Column(scale=2):
                tab5_md = gr.Markdown()
                tab5_vec = gr.Plot(label="Входной вектор")
                tab5_secret = gr.Textbox(
                    label="Ваш оригинальный секрет (protected_secret)",
                    interactive=False,
                    type="password",
                    elem_classes=["secret-field"]
                )
                tab5_reveal = gr.Button("👁 Показать секрет", size="sm")
                tab5_secret_plain = gr.Textbox(
                    label="Секрет (открытый вид — только для проверки)",
                    interactive=False,
                    visible=False
                )
                tab5_status = gr.Markdown()

        def toggle_tab5(m):
            is_dataset = (m == "Из датасета (тот же спикер)")
            return gr.update(visible=not is_dataset), gr.update(visible=is_dataset)
        tab5_mode.change(toggle_tab5, inputs=[tab5_mode], outputs=[tab5_audio, tab5_dataset_phrase])

        # При выборе НБК — подгружаем фразы привязанного спикера
        tab5_nbk.change(
            get_dataset_phrases_for_nbk,
            inputs=[tab5_nbk],
            outputs=[tab5_dataset_phrase, tab5_speaker_info]
        )

        tab5_refresh.click(lambda: gr.update(choices=get_nbk_records()), outputs=[tab5_nbk])
        tab5_btn.click(
            recover_key,
            inputs=[tab5_nbk, tab5_mode, tab5_audio, tab5_dataset_phrase],
            outputs=[tab5_md, tab5_vec, tab5_secret, tab5_status]
        )

        def reveal_secret(secret):
            return gr.update(visible=True, value=secret)
        tab5_reveal.click(reveal_secret, inputs=[tab5_secret], outputs=[tab5_secret_plain])

    # ════════════════════════════════════════════════════════════════════════
    gr.Markdown("""
---
📜 **Dasha v2.5** | ГОСТ Р 52633.5-2011 | global_minmax_abs | Кузнечик (ГОСТ Р 34.12-2015)
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )