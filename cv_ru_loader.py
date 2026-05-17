#!/usr/bin/env python3
"""
CV Ru Loader v2.32 — с поддержкой выбора фраз и метаданных для Dasha v2

Добавлено:
- get_phrases_for_speaker(speaker_id, limit) — возвращает список фраз с текстом
- get_foreign_speakers_info(exclude_speaker, max_speakers=8) — для базы "Чужой"
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


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
        # low_memory=False чтобы избежать предупреждения
        return pd.read_csv(self.validated_tsv, sep="\t", low_memory=False)

    def load_speakers_with_audio(self) -> Dict[str, List[str]]:
        """ Возвращает {client_id: [полный_путь_к_аудио, ...]} только для спикеров с >=3 файлами."""
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
                    speakers[str(client_id)] = audio_paths[:15]

            if not speakers:
                raise ValueError("В датасете нет ни одного спикера с минимум 3 аудиофайлами.")

            print(f"[CVRuLoader] ✅ Загружено {len(speakers)} реальных спикеров")
            return speakers
        except Exception as e:
            print(f"[CVRuLoader] ❌ Критическая ошибка: {e}")
            raise

    def get_phrases_for_speaker(self, speaker_id: str, limit: int = 12) -> List[Dict]:
        """ Возвращает список {'path': str, 'sentence': str, 'client_id': str} для выбора фразы"""
        try:
            df = self._load_data()
            group = df[df["client_id"] == speaker_id]
            phrases = []
            for _, row in group.head(limit).iterrows():
                audio_rel = row.get("path")
                sentence = str(row.get("sentence", "[фраза не указана]"))[:80]
                if pd.notna(audio_rel):
                    full_path = str(self.clips_dir / str(audio_rel))
                    if Path(full_path).exists():
                        phrases.append({
                            "path": full_path,
                            "sentence": sentence,
                            "client_id": str(speaker_id)
                        })
            return phrases
        except Exception as e:
            print(f"[CVRuLoader] Ошибка get_phrases: {e}")
            return []

    def get_foreign_speakers_info(self, exclude_speaker: Optional[str] = None, max_speakers: int = 8) -> List[Dict]:
        """ Возращает информацию о спикерах для базы 'Чужой' (рекомендуется по ГОСТ)"""
        try:
            df = self._load_data()
            speakers_info = []
            for client_id, group in df.groupby("client_id"):
                if exclude_speaker and str(client_id) == exclude_speaker:
                    continue
                if len(group) >= 5:
                    speakers_info.append({
                        "speaker_id": str(client_id),
                        "num_phrases": len(group),
                        "sample_sentence": str(group.iloc[0].get("sentence", ""))[:60]
                    })
                if len(speakers_info) >= max_speakers:
                    break
            return speakers_info
        except:
            return []


def load_speakers_with_audio() -> Dict[str, List[str]]:
    loader = CVRuLoader()
    return loader.load_speakers_with_audio()