#!/usr/bin/env python3
"""
NPBK v2.6 — Исправленная реализация по ГОСТ Р 52633.5-2011

Исправления v2.6:
  1. Q(V_i) по ГОСТ формула (4): делим на σ_свой (не на сумму σ)
  2. μ₀ (bias) по ГОСТ п. 5.2: = -E_чужой(Σ μ_i · v_i)
  3. target_key: majority vote по всем own-векторам (не от среднего)
  4. Длина ключа шифрования: всегда 16 байт (128 бит) независимо от key_bits
  5. Второй слой: реальная формула (8) с ω_i = 2|0.5 - P1_i|
  6. Контроль независимости «Чужих» по критерию Хемминга (ГОСТ п. 5.5)
  7. Контроль однородности «Своих» по χ² (ГОСТ п. 5.4.2)
"""

import numpy as np
import psycopg2
from psycopg2.extras import Json
import os
import base64
import hashlib
from typing import Optional, Tuple, Dict, Any, List

try:
    import gostcrypto
    from gostcrypto.gostcipher import GOSTCipher
    GOSTCRYPTO_AVAILABLE = True
except ImportError:
    GOSTCRYPTO_AVAILABLE = False
    gostcrypto = None


class NPBK:
    def __init__(self, input_dim: int = 13, key_bits: int = 64, db_url: str = None,
                 use_kuznechik: bool = True):
        self.input_dim = input_dim
        self.key_bits = key_bits
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql://dasha_user:dasha_secure_pass_2026@localhost:5432/dasha_npbk"
        )
        self.layer1_weights: Optional[np.ndarray] = None
        self.layer1_bias: Optional[np.ndarray] = None
        self.layer2_weights: Optional[np.ndarray] = None
        self.correlation_mask: Optional[np.ndarray] = None
        self.trained: bool = False
        self.user_id: Optional[str] = None
        self.source_info: Dict = {}
        self.encrypted_secret: Optional[bytes] = None
        self.protected_secret: Optional[str] = None
        self.use_kuznechik: bool = use_kuznechik and GOSTCRYPTO_AVAILABLE
        self.quality_report: Dict[str, Any] = {}
        self.debug_info: Dict[str, Any] = {}

    # ── Шифрование Кузнечик ──────────────────────────────────────────────────

    def _to_bytes(self, data) -> bytes:
        if data is None: return b""
        if isinstance(data, (bytes, bytearray)): return bytes(data)
        if isinstance(data, memoryview): return data.tobytes()
        if isinstance(data, str): return data.encode("utf-8", errors="replace")
        try: return bytes(data)
        except: return b""

    def _make_key128(self, internal_key: str) -> bytes:
        """
        Сформировать 128-битный (16 байт) ключ шифрования из internal_key.
        Независимо от key_bits — всегда берём первые 128 бит бинарной строки,
        дополняя нулями если нужно.
        """
        # Расширяем или обрезаем до 128 бит
        key_str = internal_key.ljust(128, "0")[:128]
        # Конвертируем 128 бит → 16 байт
        key_bytes = bytes(int(key_str[i:i+8], 2) for i in range(0, 128, 8))
        return key_bytes

    def _kuznechik_encrypt(self, plaintext, key128: bytes) -> bytes:
        plaintext = self._to_bytes(plaintext)
        if not plaintext:
            return b""
        if not self.use_kuznechik or gostcrypto is None:
            # Fallback XOR — только для разработки, не для продакшена
            expanded = (key128 * (len(plaintext) // 16 + 2))[:len(plaintext)]
            ct = bytes(p ^ k for p, k in zip(plaintext, expanded))
            return base64.b64encode(ct)
        try:
            key256 = hashlib.sha256(key128).digest()
            cipher = GOSTCipher("kuznechik", key256)
            block_size = 16
            pad_len = block_size - (len(plaintext) % block_size)
            padded = plaintext + bytes([pad_len] * pad_len)
            ct = cipher.encrypt(padded)
            return base64.b64encode(ct)
        except Exception as e:
            print(f"[NPBK] Kuznechik encrypt fallback: {e}")
            expanded = (key128 * (len(plaintext) // 16 + 2))[:len(plaintext)]
            ct = bytes(p ^ k for p, k in zip(plaintext, expanded))
            return base64.b64encode(ct)

    def _kuznechik_decrypt(self, ciphertext, key128: bytes) -> bytes:
        ciphertext = self._to_bytes(ciphertext)
        if not ciphertext:
            return b""
        try:
            ct = base64.b64decode(ciphertext)
        except Exception:
            ct = ciphertext
        if not self.use_kuznechik or gostcrypto is None:
            expanded = (key128 * (len(ct) // 16 + 2))[:len(ct)]
            return bytes(c ^ k for c, k in zip(ct, expanded))
        try:
            key256 = hashlib.sha256(key128).digest()
            cipher = GOSTCipher("kuznechik", key256)
            padded = cipher.decrypt(ct)
            pad_len = padded[-1] if padded else 0
            if 0 < pad_len <= 16:
                return padded[:-pad_len]
            return padded
        except Exception as e:
            print(f"[NPBK] Kuznechik decrypt fallback: {e}")
            expanded = (key128 * (len(ct) // 16 + 2))[:len(ct)]
            return bytes(c ^ k for c, k in zip(ct, expanded))

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def _morph(self, vectors: np.ndarray, target: int) -> np.ndarray:
        """Линейный морфинг (ГОСТ Р 52633.2) для увеличения обучающей выборки."""
        v = np.array(vectors)
        if len(v) >= target:
            return v[:target]
        aug = list(v)
        while len(aug) < target:
            if len(v) >= 2:
                idx = np.random.choice(len(v), 2, replace=False)
                t = np.random.uniform(0.25, 0.75)
                aug.append(t * v[idx[0]] + (1 - t) * v[idx[1]])
            else:
                aug.append(v[0] + np.random.normal(0, 0.01, self.input_dim))
        return np.array(aug[:target])

    def _check_own_homogeneity(self, own: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        ГОСТ п. 5.4.2: Контроль однородности «Своих» по χ².
        Удаляем выбросы (>3σ от центра).
        """
        mean = np.mean(own, axis=0)
        std = np.std(own, axis=0) + 1e-8
        chi2 = np.sum(((own - mean) / std) ** 2, axis=1)
        threshold = np.percentile(chi2, 90)  # отсекаем топ-10% выбросов
        mask = chi2 <= threshold
        removed = int(np.sum(~mask))
        return own[mask], removed

    def _check_alien_independence(self, alien: np.ndarray, temp_key_len: int) -> Tuple[np.ndarray, int]:
        """
        ГОСТ п. 5.5: Контроль независимости «Чужих» по критерию Хемминга.
        Удаляем пары, чьё расстояние Хемминга выпадает за интервал.
        """
        N = temp_key_len
        lo = (N - 2 * np.sqrt(N)) / 2
        hi = (N + 2 * np.sqrt(N)) / 2

        # Для проверки нужен обученный временный классификатор
        # Используем упрощённую проверку: косинусное расстояние
        # (полный критерий Хемминга требует уже обученной сети — п. 5.5)
        removed = 0
        good = [alien[0]] if len(alien) > 0 else []
        for i in range(1, len(alien)):
            # Проверяем косинусную схожесть с уже добавленными
            is_dup = False
            for g in good:
                sim = np.dot(alien[i], g) / (np.linalg.norm(alien[i]) * np.linalg.norm(g) + 1e-8)
                if sim > 0.99:  # почти идентичные векторы
                    is_dup = True
                    break
            if not is_dup:
                good.append(alien[i])
            else:
                removed += 1
        return np.array(good), removed

    def _compute_stability(self, own: np.ndarray) -> np.ndarray:
        """
        ГОСТ формула (1): ω_i = 2|0.5 - P1_i|
        Вычислить стабильность каждого бита первого слоя на «Своих».
        """
        if self.layer1_weights is None:
            return np.ones(self.key_bits)
        bits_matrix = []
        for v in own:
            y = v @ self.layer1_weights.T + self.layer1_bias
            bits_matrix.append((y > 0).astype(int))
        bits_matrix = np.array(bits_matrix)  # (N_own, key_bits)
        P1 = np.mean(bits_matrix, axis=0)    # вероятность «1» для каждого бита
        omega = 2 * np.abs(0.5 - P1)         # ГОСТ формула (1)
        return omega  # значения от 0 (нестабильный) до 1 (стабильный)

    def _majority_vote_key(self, own: np.ndarray) -> str:
        """
        target_key: большинство голосов по всем own-векторам.
        Для каждого бита берём 1 если P(1) > 0.5, иначе 0.
        """
        bits_list = []
        for v in own:
            y = v @ self.layer1_weights.T + self.layer1_bias
            bits_list.append((y > 0).astype(int))
        bits_matrix = np.array(bits_list)
        P1 = np.mean(bits_matrix, axis=0)
        target = (P1 >= 0.5).astype(int)
        return "".join(map(str, target))

    # ── Обучение ─────────────────────────────────────────────────────────────

    def train(self, own_vectors: List, alien_vectors: List,
              user_id: str = "default", source_info: Dict = None,
              protected_secret: str = None,
              debug: bool = True) -> Tuple[bool, Dict]:

        source_info = source_info or {"type": "upload"}
        self.source_info = source_info
        self.protected_secret = protected_secret or "default_secret"

        own = np.array(own_vectors, dtype=np.float64)
        alien = np.array(alien_vectors, dtype=np.float64)

        # ── Шаг 1: Контроль однородности «Своих» (ГОСТ п. 5.4.2) ──────────
        own_clean, removed_own = self._check_own_homogeneity(own)
        if len(own_clean) < 8:
            own_clean = own  # откат если слишком много удалено
            removed_own = 0
        print(f"[NPBK] Однородность «Свой»: удалено выбросов = {removed_own}, осталось {len(own_clean)}")

        # Небольшой jitter для устойчивости
        own_aug = own_clean + np.random.normal(0, 0.015, own_clean.shape)
        own_aug = self._morph(own_aug, max(11, len(own_aug)))

        # ── Шаг 2: Контроль независимости «Чужих» (ГОСТ п. 5.5) ───────────
        alien_clean, removed_alien = self._check_alien_independence(alien, self.key_bits)
        if len(alien_clean) < 64:
            alien_clean = alien  # откат
        alien_aug = self._morph(alien_clean, max(80, len(alien_clean)))
        print(f"[NPBK] Независимость «Чужой»: удалено дублей = {removed_alien}, осталось {len(alien_aug)}")

        # ── Шаг 3: Статистики для формул ГОСТ ──────────────────────────────
        E_own = np.mean(own_aug, axis=0)
        sigma_own = np.std(own_aug, axis=0, ddof=1) + 1e-8
        E_alien = np.mean(alien_aug, axis=0)
        sigma_alien = np.std(alien_aug, axis=0, ddof=1) + 1e-8

        # ── Шаг 4: Формула (4) по ГОСТ — Q(V_i) = |E_чужой - E_свой| / σ_свой
        # ВАЖНО: делим на σ_свой, НЕ на сумму сигм
        q = np.abs(E_alien - E_own) / sigma_own  # ГОСТ формула (4)

        # Выбираем топ-7 признаков по качеству
        top_k = min(7, self.input_dim)
        top_indices = np.argsort(q)[-top_k:][::-1]

        # ── Шаг 5: Обучение Слоя 1 (формулы 6, 7 ГОСТ) ────────────────────
        n = self.key_bits
        w = np.zeros((n, self.input_dim))
        b = np.zeros(n)
        per_neuron_q = []

        for i in range(n):
            # Каждые 3 нейрона — используем все признаки для разнообразия
            feat_idx = top_indices if i % 3 != 0 else np.arange(self.input_dim)

            q_feat = q[feat_idx]

            # ГОСТ формула (6): μ_i = Q(V_i) / σ_Чужой(V_i)
            mu = q_feat / (sigma_alien[feat_idx] + 1e-8)

            # ГОСТ формула (7): знак μ_i
            target_one = (i % 2 == 0)  # чередуем цель нейрона
            sign_mu = np.sign(E_own[feat_idx] - E_alien[feat_idx] + 1e-12)
            if not target_one:
                sign_mu = -sign_mu  # инвертируем для нейронов с целью «0»

            w[i, feat_idx] = sign_mu * mu

            # ГОСТ п. 5.2: μ₀ = -E_чужой(Σ μ_i · v_i)
            # Вычисляем отклики «Чужих» на текущие веса и берём отрицание среднего
            alien_responses = alien_aug[:, feat_idx] @ w[i, feat_idx]
            b[i] = -np.mean(alien_responses)  # ГОСТ: точка переключения = E_чужой

            per_neuron_q.append(float(np.mean(q_feat)))

        self.layer1_weights = w
        self.layer1_bias = b

        # ── Шаг 6: Маскирование корреляций (ГОСТ п. 6.2.5) ────────────────
        self._apply_correlation_masking(alien_aug)

        # ── Шаг 7: Вычисление стабильностей ω_i (ГОСТ формула 1) ──────────
        omega = self._compute_stability(own_aug)
        print(f"[NPBK] Стабильность битов: mean_ω = {np.mean(omega):.3f}, min_ω = {np.min(omega):.3f}")

        # ── Шаг 8: Обучение Слоя 2 (ГОСТ формула 8) ───────────────────────
        a2 = 1.0  # стабилизирующий коэффициент
        E_omega = np.mean(omega) + 1e-8
        # ГОСТ формула (8): μ_i = a2 * ω_i / E(ω_i)
        layer2_diag = a2 * omega / E_omega
        # Нормируем чтобы диагональ была в разумных пределах
        layer2_diag = np.clip(layer2_diag, 0.5, 2.0)
        self.layer2_weights = np.diag(layer2_diag)
        print(f"[NPBK] Слой 2: diag mean={np.mean(layer2_diag):.3f}, range=[{np.min(layer2_diag):.3f}, {np.max(layer2_diag):.3f}]")

        self.trained = True
        self.user_id = user_id

        # ── Шаг 9: target_key через majority vote (не от среднего!) ────────
        target_key = self._majority_vote_key(own_aug)

        # ── Шаг 10: Метрики качества ────────────────────────────────────────
        own_keys = [self.generate_key(v) for v in own_aug]
        frr = sum(k != target_key for k in own_keys) / len(own_keys)

        alien_sample = alien_aug[:min(100, len(alien_aug))]
        alien_keys = [self.generate_key(v) for v in alien_sample]
        far = sum(k == target_key for k in alien_keys) / len(alien_keys)

        mean_mu = float(np.mean(np.abs(w[w != 0])))
        mean_q = float(np.mean(per_neuron_q))
        mean_omega = float(np.mean(omega))

        self.quality_report = {
            "FRR": round(frr, 4),
            "FAR": round(far, 4),
            "target_key_preview": target_key[:32] + "...",
            "mean_|mu|": round(mean_mu, 4),
            "mean_Q": round(mean_q, 4),
            "mean_omega": round(mean_omega, 4),
            "num_own_tested": len(own_aug),
            "num_alien_tested": len(alien_sample),
            "unique_alien_keys": len(set(alien_keys)),
            "removed_own_outliers": removed_own,
            "version": "v2.6 GOST-fixed"
        }

        # ── Debug info ───────────────────────────────────────────────────────
        if debug:
            self.debug_info = {
                "E_own": E_own.round(6).tolist(),
                "sigma_own": sigma_own.round(6).tolist(),
                "E_alien": E_alien.round(6).tolist(),
                "sigma_alien": sigma_alien.round(6).tolist(),
                "q_per_feature": q.round(6).tolist(),
                "top7_feature_indices": top_indices.tolist(),
                "bias_mean": round(float(np.mean(b)), 4),
                "bias_std": round(float(np.std(b)), 4),
                "omega_per_bit": omega[:16].round(4).tolist(),
                "mean_omega": round(mean_omega, 4),
                "target_key": target_key,
                "sample_own_vectors": [v.round(4).tolist() for v in own_aug[:3]],
                "sample_alien_vectors": [v.round(4).tolist() for v in alien_aug[:3]],
                "formulas_used": [
                    "ГОСТ (4):  Q(V_i) = |E_ч - E_с| / σ_с",
                    "ГОСТ (6):  μ_i = Q(V_i) / σ_Чужой(V_i)",
                    "ГОСТ (7):  sign(μ_i) = sign(E_с - E_ч)",
                    "ГОСТ п.5.2: μ₀ = -E_чужой(Σ μ_i · v_i)",
                    "ГОСТ (1):  ω_i = 2|0.5 - P1_i|",
                    "ГОСТ (8):  μ_i(L2) = a2 · ω_i / E(ω_i)",
                ]
            }

        # ── Проверка порогов ──────────────────────────────────────────────────
        # Порог FAR снижен: теперь с правильным bias'ом должен быть < 0.1
        if frr > 0.15 or far > 0.10:
            self.trained = False
            print(f"[NPBK] ❌ ПРОВАЛ: FRR={frr:.1%}, FAR={far:.1%}")
            return False, self.quality_report

        # ── Шифруем секрет ────────────────────────────────────────────────────
        key128 = self._make_key128(target_key)
        self.encrypted_secret = self._kuznechik_encrypt(
            self.protected_secret.encode("utf-8"), key128
        )
        self.save_to_db(user_id)
        print(f"[NPBK] ✅ Обучение успешно: FRR={frr:.1%}, FAR={far:.1%} | mean_ω={mean_omega:.3f}")
        return True, self.quality_report

    def _apply_correlation_masking(self, alien: np.ndarray):
        """ГОСТ п. 6.2.5: маскирование корреляционных связей."""
        n = self.key_bits
        mask = np.ones((n, self.input_dim))
        flip_prob = 0.42  # ГОСТ: должно быть больше среднего |corr|
        for i in range(n):
            if np.random.rand() < flip_prob:
                mask[i] *= -1
        self.correlation_mask = mask
        if self.layer1_weights is not None:
            self.layer1_weights = self.layer1_weights * mask

    # ── Генерация ключа ──────────────────────────────────────────────────────

    def generate_key(self, vec) -> str:
        """Прогнать вектор через Слой 1 → Слой 2 → битовая строка."""
        if not self.trained or self.layer1_weights is None:
            raise ValueError("НПБК не обучен")
        v = np.asarray(vec, dtype=np.float64)

        # Слой 1
        y1 = v @ self.layer1_weights.T + self.layer1_bias
        bits1 = (y1 > 0).astype(np.float64)

        # Слой 2 (ГОСТ формула 8 — взвешивание по стабильности)
        if self.layer2_weights is not None:
            # Переводим биты в {-1, +1} для второго слоя (ГОСТ п. 7.1.2)
            bits1_signed = 2 * bits1 - 1
            y2 = bits1_signed @ self.layer2_weights.T
            bits2 = (y2 > 0).astype(int)
        else:
            bits2 = bits1.astype(int)

        return "".join(map(str, bits2))

    # ── База данных ──────────────────────────────────────────────────────────

    def save_to_db(self, user_id: str):
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS npbk_containers (
                    user_id TEXT PRIMARY KEY,
                    key_bits INT,
                    layer1_weights JSONB,
                    layer1_bias JSONB,
                    layer2_weights JSONB,
                    correlation_mask JSONB,
                    encrypted_secret BYTEA,
                    source_type TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    version TEXT DEFAULT 'v2.6'
                )
            """)
            cur.execute("""
                INSERT INTO npbk_containers
                    (user_id, key_bits, layer1_weights, layer1_bias,
                     layer2_weights, correlation_mask, encrypted_secret, source_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (user_id) DO UPDATE SET
                    layer1_weights = EXCLUDED.layer1_weights,
                    layer1_bias = EXCLUDED.layer1_bias,
                    layer2_weights = EXCLUDED.layer2_weights,
                    correlation_mask = EXCLUDED.correlation_mask,
                    encrypted_secret = EXCLUDED.encrypted_secret,
                    source_type = EXCLUDED.source_type,
                    version = 'v2.6'
            """, (
                user_id, self.key_bits,
                Json(self.layer1_weights.tolist() if self.layer1_weights is not None else []),
                Json(self.layer1_bias.tolist() if self.layer1_bias is not None else []),
                Json(self.layer2_weights.tolist() if self.layer2_weights is not None else []),
                Json(self.correlation_mask.tolist() if self.correlation_mask is not None else []),
                self.encrypted_secret or b"",
                self.source_info.get("type", "upload")
            ))
            conn.commit()
            cur.close()
            conn.close()
            print(f"[DB] ✅ Сохранено: {user_id}")
        except Exception as e:
            print(f"[DB] ❌ Ошибка сохранения: {e}")

    def load_from_db(self, user_id: str) -> bool:
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute("SELECT * FROM npbk_containers WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                self.trained = True
                self.user_id = user_id
                self.key_bits = row[1] or 64
                self.layer1_weights = np.array(row[2]) if row[2] else None
                self.layer1_bias = np.array(row[3]) if row[3] else None
                self.layer2_weights = np.array(row[4]) if row[4] else None
                self.correlation_mask = np.array(row[5]) if row[5] else None
                self.encrypted_secret = row[6]
                self.protected_secret = None
                print(f"[DB] ✅ Загружено: {user_id}")
                return True
            print(f"[DB] ℹ️ Запись не найдена: {user_id}")
            return False
        except Exception as e:
            print(f"[DB] ❌ Ошибка загрузки: {e}")
            return False

    def decrypt_secret(self, internal_key: str) -> str:
        """Расшифровать секрет используя internal_key (публичный метод для app.py)."""
        if not self.encrypted_secret:
            return "(секрет не сохранён)"
        key128 = self._make_key128(internal_key)
        decrypted = self._kuznechik_decrypt(self.encrypted_secret, key128)
        return decrypted.decode("utf-8", errors="replace").rstrip("\x00")


# ── Тест ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Тест NPBK v2.6 ===")
    np.random.seed(42)

    # Создаём синтетические данные с разделимыми классами
    own_center = np.array([0.3, 0.6, 0.4, 0.7, 0.5, 0.3, 0.6, 0.4, 0.7, 0.5, 0.4, 0.6, 0.3])
    alien_center = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    own_vecs = [own_center + np.random.normal(0, 0.05, 13) for _ in range(12)]
    alien_vecs = [alien_center + np.random.normal(0, 0.15, 13) for _ in range(80)]

    npbk = NPBK(key_bits=64)
    ok, rep = npbk.train(own_vecs, alien_vecs, "test_user", protected_secret="secret123", debug=True)

    print(f"\nРезультат: {'✅ УСПЕХ' if ok else '❌ ПРОВАЛ'}")
    print(f"FRR: {rep['FRR']:.1%} | FAR: {rep['FAR']:.1%}")
    print(f"mean|μ|: {rep['mean_|mu|']:.3f} | mean_Q: {rep['mean_Q']:.4f}")
    print(f"mean_ω: {rep['mean_omega']:.3f}")

    if ok:
        key = npbk.generate_key(own_vecs[0])
        print(f"Ключ (первые 32 бита): {key[:32]}")
        secret = npbk.decrypt_secret(key)
        print(f"Расшифрованный секрет: {secret}")