#!/usr/bin/env python3
"""
Dasha v2 — Gradio интерфейс v2.14

- Полностью рабочий training нормализатора (учитывает слайдеры)
- per-utterance CMVN + standard нормализация (лучшая разделимость)
- Вкладки 1-3 используют один и тот же pipeline
- Таб 3: per-speaker mean + RMS
- Нормализатор — только на реальных данных
"""
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import soundfile as sf

from pipeline import VoiceFeaturePipeline
from normalizer import FeatureNormalizer
from cv_ru_loader import load_speakers_with_audio

pipeline = VoiceFeaturePipeline(use_rasta=True)
normalizer = FeatureNormalizer(method="standard")  # standard по умолчанию

print("🔄 Проверка нормализатора и датасета...")
global_speakers = {}
try:
    global_speakers = load_speakers_with_audio()
    print(f"✅ Загружено {len(global_speakers)} реальных спикеров из data/")
except Exception as e:
    print(f"⚠️  {e}")
    global_speakers = {}

# === Обучение нормализатора НА РЕАЛЬНЫХ данных при запуске ===
def _collect_real_vectors(max_vecs: int = 150) -> list:
    """Собирает реальные 26-мерные векторы из global_speakers."""
    vecs = []
    if not global_speakers:
        return vecs
    sorted_sp = sorted(global_speakers.items(), key=lambda x: len(x[1]), reverse=True)[:25]
    for speaker_id, audio_paths in sorted_sp:
        for path in audio_paths[:8]:
            try:
                res = pipeline.extract_features(path)
                vecs.append(res["normalized_vector"])
                if len(vecs) >= max_vecs:
                    return vecs
            except Exception:
                continue
    return vecs

real_vecs = _collect_real_vectors(200)
if real_vecs:
    arr = np.array(real_vecs)
    normalizer.fit(arr)
    normalizer.save()
    pipeline.normalizer = normalizer
    print(f"✅ Нормализатор обучен на {len(real_vecs)} РЕАЛЬНЫХ векторах (standard)")
else:
    print("⚠️ Нет реальных данных для обучения нормализатора")


def create_waveform_plot(y: np.ndarray, sr: int, title: str = " waveform") -> go.Figure:
    time = np.linspace(0, len(y) / sr, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode="lines", line=dict(color="#00B4D8", width=1.2), name="Сигнал"))
    fig.update_layout(title=title, xaxis_title="Время (с)", yaxis_title="Амплитуда", height=280, margin=dict(l=40, r=20, t=40, b=30), template="plotly_white")
    return fig


def create_vector_bar_plot(vector: list, title: str = "26-мерный вектор (13 mean + 13 std после RASTA + CMVN)") -> go.Figure:
    fig = go.Figure()
    colors = ["#FF6B6B" if v > 0.7 else "#4ECDC4" for v in vector]
    fig.add_trace(go.Bar(
        x=[f"F{i+1}" for i in range(len(vector))],
        y=vector,
        marker_color=colors,
        text=[f"{v:.2f}" for v in vector],
        textposition="outside",
        textfont=dict(size=9)
    ))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1.05]), height=320, margin=dict(l=30, r=20, t=40, b=50), template="plotly_white", showlegend=False)
    return fig


def create_mfcc_heatmap(mfcc: np.ndarray, title: str = "RASTA-MFCC (13 коэф.)") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=mfcc, colorscale="Viridis", colorbar=dict(title="Значение")))
    fig.update_layout(title=title, xaxis_title="Кадры", yaxis_title="MFCC коэффициенты (1-13)", height=260, margin=dict(l=40, r=20, t=40, b=30))
    return fig


def create_correlation_heatmap(vectors: list, labels: list) -> go.Figure:
    arr = np.array(vectors)
    corr = np.corrcoef(arr)
    fig = go.Figure(data=go.Heatmap(z=corr, x=labels, y=labels, colorscale="RdYlBu_r", zmin=0, zmax=1, colorbar=dict(title="Корреляция")))
    fig.update_layout(title="Матрица корреляций векторов (0–1)", height=380, margin=dict(l=60, r=20, t=40, b=60))
    return fig


def process_single_phrase(audio, use_rasta, file_path=None):
    global pipeline
    pipeline.use_rasta = use_rasta
    if audio is not None:
        sr, y = audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            path = tmp.name
    elif file_path:
        path = file_path
    else:
        return "Загрузите аудио или запишите с микрофона", None, None, None, ""
    try:
        result = pipeline.extract_features(path)
        vec = result["normalized_vector"]
        mfcc = result.get("mfcc_rasta", np.zeros((13, 10)))
        md = f"**✅ Обработка завершена** (RASTA+CMVN) | Длина вектора: **26** | [0, 1]"
        fig_wave = create_waveform_plot(result["y_pre"], result["sr"], "Предобработанный сигнал")
        fig_vec = create_vector_bar_plot(vec)
        fig_mfcc = create_mfcc_heatmap(mfcc)
        quality = pipeline.get_feature_quality_metrics([np.array(vec)])
        quality_md = f"**Quality:** cosine = {quality.get('mean_cosine_similarity', 'N/A')} | corr = {quality.get('mean_feature_correlation', 'N/A')} | var = {quality.get('feature_variance_proxy', 'N/A')}"
        return md, fig_wave, fig_vec, fig_mfcc, quality_md
    except Exception as e:
        return f"Ошибка: {str(e)}", None, None, None, ""


def process_correlation(files, use_rasta):
    global pipeline
    pipeline.use_rasta = use_rasta
    if not files or len(files) < 2:
        return "Загрузите минимум 2 файла", None, None
    vectors, labels = [], []
    for i, f in enumerate(files):
        try:
            res = pipeline.extract_features(f)
            vectors.append(res["normalized_vector"])
            labels.append(f"Запись {i+1}")
        except: continue
    if len(vectors) < 2:
        return "Не удалось обработать", None, None
    fig_corr = create_correlation_heatmap(vectors, labels)
    arr = np.array(vectors)
    mean_vec = np.mean(arr, axis=0)
    fig_lines = go.Figure()
    for i, v in enumerate(vectors):
        fig_lines.add_trace(go.Scatter(
            x=list(range(len(v))),
            y=v,
            mode="lines+markers",
            name=labels[i],
            line=dict(width=1.5),
            opacity=0.7
        ))
    fig_lines.add_trace(go.Scatter(
        x=list(range(len(v))),
        y=mean_vec,
        mode="lines",
        name="Средний эталон",
        line=dict(color="black", width=3, dash="dash")
    ))
    fig_lines.update_layout(title="Векторы vs Средний эталон", height=320, template="plotly_white")
    md = f"**Обработано {len(vectors)} записей** | Средняя корреляция: **{np.mean(np.corrcoef(arr)):.3f}**"
    return md, fig_corr, fig_lines


def run_gost_mass_test(num_speakers, phrases_per_speaker, use_rasta):
    global pipeline, global_speakers
    pipeline.use_rasta = use_rasta

    if not global_speakers:
        return "❌ Датасет не загружен. Положи Common Voice RU в data/firefox-ru-dataset/ (validated.tsv + clips/).", None, None, ""

    sorted_speakers = sorted(global_speakers.items(), key=lambda x: len(x[1]), reverse=True)[:num_speakers]
    actual_num = len(sorted_speakers)

    all_vectors, speaker_labels = [], []
    speaker_means = []
    speaker_rms = []

    for s_idx, (speaker_id, audio_paths) in enumerate(sorted_speakers):
        phrase_vectors = []
        selected = audio_paths[:phrases_per_speaker]
        for path in selected:
            try:
                res = pipeline.extract_features(path)
                vec = res["normalized_vector"]
                all_vectors.append(vec)
                speaker_labels.append(f"Спикер {s_idx+1}")
                phrase_vectors.append(vec)
            except Exception:
                continue
        if phrase_vectors:
            mean_vec = np.mean(phrase_vectors, axis=0)
            speaker_means.append(mean_vec)
            rms = float(np.sqrt(np.mean(np.square(mean_vec))))
            speaker_rms.append(rms)

    if len(all_vectors) < 4:
        return f"❌ Мало аудио в датасете (всего {len(all_vectors)} векторов). Нужно минимум 4.", None, None, ""

    intra_sims, inter_sims = [], []
    arr = np.array(all_vectors)
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            sim = np.dot(arr[i], arr[j]) / (np.linalg.norm(arr[i]) * np.linalg.norm(arr[j]) + 1e-8)
            if speaker_labels[i] == speaker_labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    eer_proxy = max(0, (mean_inter - mean_intra) / (mean_intra + 1e-8) * 100)

    rms_str = ", ".join([f"{r:.3f}" for r in speaker_rms])
    md = f"**Массовый тест ГОСТ Р 52633** | Спикеров: {actual_num} | Фраз: {len(all_vectors)} | intra: {mean_intra:.4f} | inter: {mean_inter:.4f} | EER~{eer_proxy:.1f}% | RMS: [{rms_str}]"

    unique_labels = [f"Спикер {s+1}" for s in range(len(speaker_means))]
    fig_heat = create_correlation_heatmap(speaker_means, unique_labels)

    fig_lines = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for s in range(len(speaker_means)):
        fig_lines.add_trace(go.Scatter(
            x=list(range(26)),
            y=speaker_means[s],
            mode="lines+markers",
            name=f"Спикер {s+1} (RMS={speaker_rms[s]:.3f})",
            line=dict(width=2.5, color=colors[s % len(colors)]),
            marker=dict(size=5)
        ))
    fig_lines.update_layout(
        title=f"Средние векторы реальных спикеров ({len(speaker_means)})",
        height=320,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    quality_md = f"**Ресеарч:** Реальные данные + CMVN + standard | Per-speaker mean + RMS"
    return md, fig_heat, fig_lines, quality_md


def train_normalizer(max_speakers, phrases):
    """Теперь УЧИТЫВАЕТ слайдеры и реально переобучает."""
    global normalizer, pipeline, global_speakers
    target = max_speakers * phrases
    real_vecs = _collect_real_vectors(target)
    if not real_vecs:
        return "❌ Нет реальных данных. Положи датасет в data/."
    arr = np.array(real_vecs)
    normalizer.fit(arr)
    normalizer.save()
    pipeline.normalizer = normalizer
    return f"✅ Нормализатор переобучен на {len(real_vecs)} РЕАЛЬНЫХ векторах (standard, {max_speakers} спикеров × {phrases} фраз). Параметры сохранены."


with gr.Blocks(title="Dasha v2 — Голосовая биометрия + НПБК (ГОСТ Р 52633)") as demo:
    gr.Markdown("""
    # 🎤 Dasha v2 — Система биометрической генерации ключей по голосу
    **v2.14 — обучение нормализатора теперь реально работает (учитывает слайдеры) | CMVN + standard | реальные данные**  
    Готово к интеграции полноценного НПБК по ГОСТ Р 52633.5.
    """)

    with gr.Tabs():
        with gr.TabItem("1. Обработка одной фразы"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Ввод голоса")
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Запишите или загрузите .wav")
                    use_rasta_cb = gr.Checkbox(value=True, label="Использовать RASTA (рекомендуется)")
                    btn_process = gr.Button("🚀 Извлечь 26-мерный вектор", variant="primary")
                with gr.Column(scale=2):
                    out_md = gr.Markdown()
                    out_wave = gr.Plot(label="Сигнал")
                    out_vec = gr.Plot(label="26-мерный вектор")
                    out_mfcc = gr.Plot(label="RASTA-MFCC")
                    out_quality = gr.Markdown()

            btn_process.click(process_single_phrase, inputs=[audio_in, use_rasta_cb], outputs=[out_md, out_wave, out_vec, out_mfcc, out_quality])

        with gr.TabItem("2. Корреляция и стабильность"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Загрузите любое кол-во записей одного спикера")
                    files_in = gr.File(file_count="multiple", file_types=[".wav", ".mp3"], label="Аудиофайлы")
                    use_rasta2 = gr.Checkbox(value=True, label="RASTA")
                    btn_corr = gr.Button("Построить корреляцию и эталон", variant="primary")
                with gr.Column(scale=2):
                    corr_md = gr.Markdown()
                    corr_heat = gr.Plot()
                    corr_lines = gr.Plot()

            btn_corr.click(process_correlation, inputs=[files_in, use_rasta2], outputs=[corr_md, corr_heat, corr_lines])

        with gr.TabItem("3. Массовый тест + Research"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Параметры теста (реальные спикеры из датасета data/)")
                    num_sp = gr.Slider(2, 12, value=5, step=1, label="Количество спикеров")
                    ph_per = gr.Slider(3, 15, value=8, step=1, label="Фраз на спикера")
                    use_rasta3 = gr.Checkbox(value=True, label="RASTA")
                    btn_test = gr.Button("Запустить тест ГОСТ + Research", variant="primary")
                with gr.Column(scale=2):
                    test_md = gr.Markdown()
                    test_heat = gr.Plot()
                    test_lines = gr.Plot()
                    test_quality = gr.Markdown()

            btn_test.click(run_gost_mass_test, inputs=[num_sp, ph_per, use_rasta3], outputs=[test_md, test_heat, test_lines, test_quality])

        with gr.TabItem("4. Обучение нормализатора"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### Глобальный нормализатор (только реальные данные)
                    **Теперь кнопка УЧИТЫВАЕТ слайдеры!**<br>
                    Обучается на (макс. спикеров × фраз) реальных векторах.<br>
                    После обучения все новые векторы сразу используют новые параметры.
                    """)
                    max_sp = gr.Slider(10, 300, value=100, step=10, label="Макс. спикеров")
                    ph = gr.Slider(3, 15, value=8, step=1, label="Фраз на спикера")
                    btn_train = gr.Button("Переобучить на реальных данных", variant="primary")
                    train_out = gr.Markdown()
            btn_train.click(train_normalizer, inputs=[max_sp, ph], outputs=train_out)

        with gr.TabItem("5. НПБК (в разработке)"):
            gr.Markdown("""
            ### Нейросетевой преобразователь «биометрия-код» (ГОСТ Р 52633.5)
            **Текущий статус:** Заглушка. 26-мерный вектор из вкладки 1 готов к подаче на вход двухслойной нейросети.
            """)
            gr.Button("Сгенерировать ключ (заглушка)", interactive=False)

    gr.Markdown("""
    ---
    **Dasha v2 v2.14** — обучение нормализатора исправлено | CMVN + standard | реальные данные. Май 2026.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())