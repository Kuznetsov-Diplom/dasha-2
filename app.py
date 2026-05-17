#!/usr/bin/env python3
"""
Dasha v2.33 — 2-ключевая система по ГОСТ Р 52633.5-2011 + ГОСТ Р 52633.2 (размножение)

Добавлено в v2.33:
- Кнопка "Показать содержимое БД" (только user_id, source_type, created_at, защищённый protected_secret)
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import tempfile
import random
import json
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

pipeline = VoiceFeaturePipeline(use_rasta=False, use_deltas=False)
npbk = NPBK(key_bits=128)
loader = CVRuLoader()
global_speakers = load_speakers_with_audio()
speaker_list = sorted([sid for sid, paths in global_speakers.items() if len(paths) >= 5])
trained_speakers = set()
TRAINED_FILE = Path("models/trained_speakers.json")
if TRAINED_FILE.exists():
    trained_speakers = set(json.load(open(TRAINED_FILE)))

def save_trained():
    TRAINED_FILE.parent.mkdir(exist_ok=True)
    json.dump(list(trained_speakers), open(TRAINED_FILE, "w"))

def get_available_speakers():
    return [s for s in speaker_list if s not in trained_speakers]

def get_random_available_speaker():
    avail = get_available_speakers()
    return random.choice(avail) if avail else None

def get_nbk_records():
    try:
        conn = psycopg2.connect(npbk.db_url)
        cur = conn.cursor()
        cur.execute("SELECT user_id, protected_secret, source_type, created_at FROM npbk_containers ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [f"{r[0]} | {r[2]} | {r[3]}" for r in rows]
    except:
        return []

def get_db_contents():
    """ Безопасный просмотр БД (показываем только несекретные поля) """
    try:
        conn = psycopg2.connect(npbk.db_url)
        cur = conn.cursor()
        cur.execute("""
            SELECT user_id, 
                   LEFT(protected_secret, 8) || '...' || RIGHT(protected_secret, 4) as secret_masked,
                   source_type, 
                   created_at 
            FROM npbk_containers 
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            return "В БД пока нет записей."
        
        md = "**Содержимое таблицы npbk_containers (masked protected_secret):**

"
        for r in rows:
            md += f"- **{r[0]}** | Секрет: `{r[1]}` | Тип: {r[2]} | {r[3]}\n"
        return md
    except Exception as e:
        return f"Ошибка подключения к БД: {e}"

def create_vector_bar_plot(vector, title="Нормализованный 13-мерный вектор [0,1]"):
    fig = go.Figure(go.Bar(x=[f"F{i+1}" for i in range(13)], y=vector, marker_color="#FF6B6B", text=[f"{v:.3f}" for v in vector], textposition="outside"))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1.05]), height=320, template="plotly_white")
    return fig

def create_binary_key_plot(binary_str, title="Internal key (НПБК)"):
    bits = [int(b) for b in binary_str[:64]]
    fig = go.Figure(go.Bar(x=list(range(len(bits))), y=bits, marker_color="#00B4D8"))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1.1]), height=180, template="plotly_white")
    return fig

def morph_augment(vectors: list, target_count: int = 11) -> list:
    if len(vectors) >= target_count:
        return vectors
    augmented = vectors.copy()
    while len(augmented) < target_count:
        if len(vectors) >= 2:
            a, b = random.sample(vectors, 2)
            alpha = random.uniform(0.2, 0.8)
            morphed = (np.array(a) * alpha + np.array(b) * (1 - alpha)).tolist()
            augmented.append(morphed)
        else:
            base = np.array(vectors[0])
            noise = np.random.normal(0, 0.02, size=13).tolist()
            augmented.append((base + noise).tolist())
    return augmented[:target_count]

def register_npbk(mode, audio_files, speaker_id, user_name, desired_key, progress=gr.Progress()):
    progress(0, desc="Подготовка...")
    if not user_name:
        user_name = speaker_id or "user_" + str(random.randint(1000,9999))

    vectors = []
    paths = []
    if mode == "Из датасета" and speaker_id:
        if speaker_id in trained_speakers:
            return "Ошибка: спикер уже обучен", None, None, None, None, None, None, None
        paths = random.sample(global_speakers[speaker_id], min(12, len(global_speakers[speaker_id])))
    elif audio_files:
        paths = audio_files if isinstance(audio_files, list) else [audio_files]
    else:
        return "Выберите источник данных", None, None, None, None, None, None, None

    for p in paths:
        try:
            res = pipeline.extract_features(p)
            vectors.append(res["normalized_vector"])
        except: continue

    if len(vectors) < 11:
        vectors = morph_augment(vectors, target_count=11)
        progress(0.2, desc=f"Размножено до 11 примеров (морфинг по ГОСТ)...")

    if len(vectors) < 8:
        return f"Мало записей даже после размножения ({len(vectors)}). Нужно минимум 8", None, None, None, None, None, None, None

    progress(0.3, desc="Генерация internal_key...")
    internal_key = ''.join(random.choice('01') for _ in range(128))

    progress(0.5, desc="Обучение НПБК (защита вашего ключа)...
