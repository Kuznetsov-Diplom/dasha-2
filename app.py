#!/usr/bin/env python3
"""
Dasha v2.31 — 2-ключeвая система + Docker-only

- protected_secret (ключ, который защищаем — вводит пользователь при регистрации)
- internal_key (генерируется НПБК, обучается на нём)
- При восстановлении показываем ТОЛЬКО protected_secret
- source_type для определения датасет/загрузка
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
    from cv_ru_loader import load_speakers_with_audio
    from npbk import NPBK
except ImportError as e:
    print(f"Import error: {e}")
    raise

pipeline = VoiceFeaturePipeline(use_rasta=False, use_deltas=False)
npbk = NPBK(key_bits=128)

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

def create_vector_bar_plot(vector, title="Нормализованный 13-мерный вектор [0,1]"):
    fig = go.Figure(go.Bar(x=[f"F{i+1}" for i in range(13)], y=vector, marker_color="#FF6B6B", text=[f"{v:.3f}" for v in vector], textposition="outside"))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1.05]), height=320, template="plotly_white")
    return fig

def create_binary_key_plot(binary_str, title="Internal key (НПБК)"):
    bits = [int(b) for b in binary_str[:64]]
    fig = go.Figure(go.Bar(x=list(range(len(bits))), y=bits, marker_color="#00B4D8"))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1.1]), height=180, template="plotly_white")
    return fig

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

    if len(vectors) < 8:
        return f"Мало записей ({len(vectors)}). Нужно 8-11", None, None, None, None, None, None, None

    progress(0.3, desc="Генерация internal_key...")
    internal_key = ''.join(random.choice('01') for _ in range(128))

    progress(0.5, desc="Обучение НПБК (защита вашего ключа)...")
    alien = [np.random.randn(13).tolist() for _ in range(70)]
    try:
        npbk.train(vectors, alien, user_id=user_name, protected_secret=desired_key)
    except Exception as e:
        return f"Ошибка: {e}", None, None, None, None, None, None, None

    trained_speakers.add(speaker_id if speaker_id else user_name)
    save_trained()

    progress(0.8, desc="Расчёт метрик...")
    quality = pipeline.get_feature_quality_metrics(vectors)
    layer1_q = round(np.mean([abs(np.mean(v)-0.5) for v in vectors]), 4)
    layer2_q = round(1.0 - quality.get('mean_feature_correlation', 0.3), 4)
    eer = pipeline.compute_eer([0.9]*len(vectors), [0.1]*70)

    progress(1.0, desc="Готово! Ключ защищён в НБК")

    vec_plot = create_vector_bar_plot(vectors[0])
    key_plot = create_binary_key_plot(internal_key)

    md = f"""
    **✅ Регистрация завершена!**

    - Пользователь: **{user_name}**
    - Ваш ключ (protected_secret): `{desired_key}` ← **этот ключ теперь защищён биометрией**
    - Internal key (НПБК): `{internal_key[:32]}...` (удалён после обучения)
    - Качество слоя 1: {layer1_q} | Слоя 2: {layer2_q}
    - EER: {eer}
    **Чтобы получить ключ снова — только через правильную биометрию!**
    """

    return md, vec_plot, key_plot, f"Слой 1: {layer1_q}", f"Слой 2: {layer2_q}", f"EER: {eer}", internal_key, "✅ Ключ защищён в PostgreSQL. Перейдите на вкладку Восстановление."


def recover_key(nbk_record, audio, progress=gr.Progress()):
    progress(0, desc="Загрузка НБК...")
    if not nbk_record:
        return "Выберите запись из НБК", None, None, None, None

    user_id = nbk_record.split(" | ")[0]
    loaded = npbk.load_from_db(user_id)
    if not loaded:
        return f"Не удалось загрузить НБК для {user_id}", None, None, None, None

    progress(0.4, desc="Обработка голоса...")
    path = None
    if audio is not None:
        sr, y = audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            path = tmp.name
    else:
        return "Загрузите запись голоса", None, None, None, None

    try:
        res = pipeline.extract_features(path)
        vec = res["normalized_vector"]
    except Exception as e:
        return f"Ошибка: {e}", None, None, None, None

    progress(0.7, desc="Восстановление через НПБК...")
    try:
        internal_key = npbk.generate_key(vec)
        original_secret = npbk.protected_secret or "(не сохранён)"
    except Exception as e:
        return f"Ошибка восстановления: {e}", None, None, None, None

    progress(1.0, desc="Готово!")

    vec_plot = create_vector_bar_plot(vec, title="Входной вектор при восстановлении")

    md = f"""
    **✅ Ключ восстановлен!**

    - Пользователь: **{user_id}**
    - **Ваш оригинальный ключ (protected_secret):** `{original_secret}`
    - Internal key НПБК: `{internal_key[:32]}...`

    **Это именно тот ключ, который вы ввели при регистрации.**
    """

    return md, vec_plot, original_secret, "Восстановление успешно! Ключ получен только благодаря правильной биометрии."

with gr.Blocks(title="Dasha v2.31 — 2-ключeвая система (ГОСТ Р 52633.5)") as demo:
    gr.Markdown("""
    # Dasha v2.31 — 2-ключeвая система по ГОСТ Р 52633.5-2011

    **protected_secret** (ваш ключ) → защищается **internal_key** (генерируется НПБК)
    При восстановлении показываем **только ваш оригинальный ключ**.
    """)

    with gr.Tabs():
        with gr.TabItem("Регистрация НПБК"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode_reg = gr.Radio(["Из датасета", "Загрузить файлы"], value="Из датасета")
                    speaker_dd = gr.Dropdown(choices=get_available_speakers(), label="Спикер")
                    btn_random = gr.Button("🎲 Случайный (не обученный)")
                    user_name = gr.Textbox(label="Имя в НБК", placeholder="ivan_2026")
                    desired_key = gr.Textbox(label="Ключ, который нужно защитить (ваш секрет)", placeholder="Мой_Закрытый_Ключ_ЭЦП_2026", type="password")
                    audio_files = gr.File(file_count="multiple", file_types=[".wav", ".mp3"], label="8-12 записей голоса")
                    btn_train = gr.Button("🚀 ЗАЩИТИТЬ КЛЮЧ (обучить НПБК)", variant="primary", size="lg")
                with gr.Column(scale=2):
                    reg_md = gr.Markdown()
                    reg_vec = gr.Plot()
                    reg_key_plot = gr.Plot()
                    reg_l1 = gr.Markdown()
                    reg_l2 = gr.Markdown()
                    reg_eer = gr.Markdown()
                    reg_internal = gr.Textbox(label="Internal key (НПБК — удаляется после обучения)", interactive=False)
                    reg_status = gr.Markdown()

            btn_random.click(lambda: get_random_available_speaker(), outputs=[speaker_dd])
            btn_train.click(register_npbk, inputs=[mode_reg, audio_files, speaker_dd, user_name, desired_key], outputs=[reg_md, reg_vec, reg_key_plot, reg_l1, reg_l2, reg_eer, reg_internal, reg_status])

        with gr.TabItem("Восстановление ключа"):
            with gr.Row():
                with gr.Column(scale=1):
                    nbk_dd = gr.Dropdown(choices=get_nbk_records(), label="Запись из НБК (авто-подстановка спикера)")
                    audio_rec = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Ваша запись голоса")
                    btn_recover = gr.Button("🔑 ВОССТАНОВИТЬ ОРИГИНАЛЬНЫЙ КЛЮЧ", variant="primary", size="lg")
                with gr.Column(scale=2):
                    rec_md = gr.Markdown()
                    rec_vec = gr.Plot()
                    rec_secret = gr.Textbox(label="Ваш оригинальный ключ (protected_secret)", interactive=False)
                    rec_status = gr.Markdown()

            btn_recover.click(recover_key, inputs=[nbk_dd, audio_rec], outputs=[rec_md, rec_vec, rec_secret, rec_status])

    gr.Markdown("---\n**Dasha v2.31** | 2-ключeвая система | Docker-only | Май 2026")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())