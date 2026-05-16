# Dasha v2 — Voice Biometric Key Generation System (GOST R 52633)

Полностью переписанная с нуля современная реализация дипломного проекта.

** Главное улучшение:** исправлена критическая ошибка v1 (нулевой вектор после CMVN), добавлена RASTA-фильтрация, глобальная нормализация и research-режим.

См. PROJECT.md для полного научного обоснования и архитектуры.

## Быстрый старт

```bash
pip install -r requirements.txt
python app.py
```

Откройте http://localhost:7860

## Структура

- `app.py` — Gradio интерфейс (5 вкладок, отличный дизайн сохранён)
- `pipeline.py` — VoiceFeaturePipeline (RASTA + правильный mean pooling + VAD)
- `normalizer.py` — FeatureNormalizer (global MinMax / Standard)
- `npbk.py` — Заглушка НПБК (готова к интеграции)
- `PROJECT.md` — Полная документация и обоснование выбора пайплайна

## Ключевые научные решения v2

- RASTA-фильтрация на MFCC (robustness к шуму и каналу)
- Mean pooling **до** нормализации (фикс бага v1)
- Глобальная нормализация на большой выборке
- 39 признаков (MFCC + Delta + Delta-Delta) — оптимально для НПБК

Готово к дальнейшей реализации полноценного НПБК и защите диплома.