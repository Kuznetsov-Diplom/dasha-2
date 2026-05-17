#!/usr/bin/env python3
"""
CV Ru Loader v2.11 — строгий загрузчик Common Voice Russian.

Только реальные данные из папки data/firefox-ru-dataset/ (validated.tsv + clips/).
Синтетика полностью удалена. Если датасет отсутствует или недостаточно спикеров — ошибка.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List


class CVRuLoader:
    def __init__(self, data_dir: str = "data/firefox-ru-dataset"):
        self.data_dir = Path(data_dir)
        self.validated_tsv = self.data_dir / "validated.tsv"
        self.clips_dir = self.data_dir / "clips"

    def _load_data(self) -> pd.DataFrame:
        if not self.validated_tsv.exists():
            raise FileNotFoundError(
                f"Файл не найден: {self.validated_tsv}\n"
                f"Положи Common Voice RU датасет в папку {self.data_dir}/\n"
                f"(validated.tsv + папка clips/ с .mp3 файлами)"
            )
        return pd.read_csv(self.validated_tsv, sep="\t")

    def load_speakers_with_audio(self) -> Dict[str, List[str]]:
        """Возвращает {client_id: [полный_путь_к_аудио, ...]} только для спикеров с >=3 файлами."""
        try:
            df = self._load_data()
            if not self.clips_dir.exists():
                raise FileNotFoundError(f"Папка с аудио не найдена: {self.clips_dir}")

            speakers: Dict[str, List[str]] = {}
            for client_id, group in df.groupby("client_id"):
                audio_paths: List[str] = []
                for _, row in group.iterrows():
                    audio_rel = row.get("path")
                    if pd.notna(audio_rel):
                        full_path = self.clips_dir / str(audio_rel)
                        if full_path.exists() and full_path.suffix.lower() in [".mp3", ".wav", ".ogg"]:
                            audio_paths.append(str(full_path))
                if len(audio_paths) >= 3:
                    # Ограничиваем 15 файлами на спикера для скорости
                    speakers[str(client_id)] = audio_paths[:15]

            if not speakers:
                raise ValueError("В датасете нет ни одного спикера с минимум 3 аудиофайлами. Проверь структуру data/firefox-ru-dataset/")

            print(f"[CVRuLoader] ✅ Загружено {len(speakers)} реальных спикеров из датасета в data/ (только реальные аудио)")
            return speakers
        except Exception as e:
            print(f"[CVRuLoader] ❌ Критическая ошибка: {e}")
            raise


def load_speakers_with_audio() -> Dict[str, List[str]]:
    loader = CVRuLoader()
    return loader.load_speakers_with_audio()