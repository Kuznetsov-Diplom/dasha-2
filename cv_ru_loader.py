#!/usr/bin/env python3
"""
CV Ru Loader — загрузчик Common Voice Russian для реальных спикеров и фраз.

Если датасет не найден — автоматически использует синтетические данные (для демо и тестов).
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List


class CVRuLoader:
    def __init__(self, data_dir: str = "data/firefox-ru-dataset"):
        self.data_dir = Path(data_dir)
        self.validated_tsv = self.data_dir / "validated.tsv"

    def _load_data(self) -> pd.DataFrame:
        if not self.validated_tsv.exists():
            raise FileNotFoundError(
                f"Файл не найден: {self.validated_tsv}\n"
                f"Скачайте Common Voice RU датасет в папку {self.data_dir}/\n"
                f"(https://commonvoice.mozilla.org/ru/datasets)"
            )
        return pd.read_csv(self.validated_tsv, sep="\t")

    def load_speakers_and_phrases(self) -> Dict[str, List[str]]:
        try:
            df = self._load_data()
            # Группируем по client_id (спикер) и берём предложения
            speakers = (
                df.groupby("client_id")["sentence"]
                .apply(lambda x: x.dropna().tolist())
                .to_dict()
            )
            # Фильтруем спикеров с хотя бы 3 фразами
            speakers = {k: v for k, v in speakers.items() if len(v) >= 3}
            print(f"[CVRuLoader] Загружено {len(speakers)} реальных спикеров из Common Voice RU")
            return speakers
        except FileNotFoundError as e:
            print(f"[CVRuLoader] {e}")
            print("[CVRuLoader] Используем СИНТЕТИЧЕСКИЕ данные для демо")
            # Синтетические спикеры (как в mass test)
            import numpy as np
            np.random.seed(42)
            synthetic = {}
            phrases = [
                "Привет, как дела?", "Сегодня хорошая погода.", "Я люблю программировать.",
                "Голосовая биометрия — это будущее.", "ГОСТ Р 52633 требует стабильности.",
                "Дашa v2 работает отлично.", "Нейросетевой преобразователь биометрия-код."
            ]
            for i in range(20):
                synthetic[f"synthetic_speaker_{i+1}"] = np.random.choice(phrases, size=5, replace=False).tolist()
            return synthetic


def load_speakers_and_phrases() -> Dict[str, List[str]]:
    loader = CVRuLoader()
    return loader.load_speakers_and_phrases()