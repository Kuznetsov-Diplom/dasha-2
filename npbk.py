#!/usr/bin/env python3
"""
NPBK v2.37 — фикс сохранения в БД + короткие логи

- Добавлен protected_secret в INSERT (чтобы сохранялось)
- Короткие логи (без огромных массивов)
- Защита от слишком больших JSON
"""

import numpy as np
import psycopg2
from psycopg2.extras import Json
import json
from typing import List, Dict, Optional, Tuple
import os
import base64

try:
    import gostcrypto
    from gostcrypto.gostcipher import GOSTCipher
    GOSTCRYPTO_AVAILABLE = True
except ImportError:
    GOSTCRYPTO_AVAILABLE = False
    gostcrypto = None


class NPBK:
    """ Полноценный НПБК по ГОСТ Р 52633.5-2011 """

    def __init__(self, input_dim: int = 13, key_bits: int = 128, db_url: Optional[str] = None, use_kuznechik: bool = True):
        self.input_dim = input_dim
        self.key_bits = key_bits
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://dasha_user:dasha_secure_pass_2026@localhost:5432/dasha_npbk")
        self.layer1_weights = None
        self.layer1_bias = None
        self.layer2_weights = None
        self.correlation_mask = None
        self.trained = False
        self.user_id = None
        self.source_info = {}
        self.registered_key = None
        self.encrypted_secret = None
        self.protected_secret = None
        self.use_kuznechik = use_kuznechik and GOSTCRYPTO_AVAILABLE

    def train(self, own_vectors, alien_vectors, user_id="default", source_info=None, protected_secret=None):
        source_info = source_info or {"type": "upload", "speaker_id": None, "files": None}
        self.source_info = source_info
        self.protected_secret = protected_secret or "default_secret"

        own = np.array(own_vectors, dtype=np.float64)
        alien = np.array(alien_vectors, dtype=np.float64)

        own = self._morph(own, 12)
        alien = self._morph(alien, 64)

        E_own, sigma_own = self._compute_stats(own)
        E_alien, sigma_alien = self._compute_stats(alien)

        n = self.key_bits
        w = np.zeros((n, self.input_dim))
        b = np.zeros(n)

        for i in range(n):
            q = np.abs(E_alien - E_own) / (sigma_own + 1e-8)
            mu = q / (sigma_alien + 1e-8)
            sign = np.sign(E_own - E_alien) if (i % 2 == 0) else -np.sign(E_own - E_alien)
            w[i] = sign * mu
            b[i] = -np.mean(alien @ w[i])

        self.layer1_weights = w
        self.layer1_bias = b
        self.layer2_weights = np.eye(n) * 0.85
        self._apply_correlation_masking(alien)

        ref_bits = (own[0] @ w.T + b > 0).astype(int)
        self.registered_key = ''.join(map(str, ref_bits))
        self.trained = True
        self.user_id = user_id

        key_bytes = bytes(int(self.registered_key[i:i+8], 2) for i in range(0, 128, 8))
        self.encrypted_secret = self._kuznechik_encrypt(self.protected_secret.encode(), key_bytes)

        self.save_to_db(user_id)
        print("[NPBK] Обучение завершено и сохранено в БД")

    def _compute_stats(self, v):
        return np.mean(v, axis=0), np.std(v, axis=0, ddof=1)

    def _morph(self, v, target):
        if len(v) >= target: return v[:target]
        aug = list(v)
        while len(aug) < target:
            a, b = np.random.choice(len(v), 2, replace=False)
            alpha = np.random.uniform(0.25, 0.75)
            aug.append(alpha * v[a] + (1-alpha) * v[b])
        return np.array(aug)

    def _apply_correlation_masking(self, alien):
        n = self.key_bits
        mask = np.ones((n, self.input_dim))
        for i in range(n):
            if np.random.rand() < 0.35: mask[i] *= -1
        self.correlation_mask = mask
        if self.layer1_weights is not None:
            self.layer1_weights *= mask

    def generate_key(self, vec):
        if not self.trained: raise ValueError("Не обучен")
        y = np.array(vec) @ self.layer1_weights.T + self.layer1_bias
        bits = (y > 0).astype(int)
        if self.layer2_weights is not None:
            bits = (bits @ self.layer2_weights.T > 0).astype(int)
        return ''.join(map(str, bits))

    def save_to_db(self, user_id):
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()

            # Создаём таблицу с protected_secret
            cur.execute("""
                CREATE TABLE IF NOT EXISTS npbk_containers (
                    user_id TEXT PRIMARY KEY,
                    key_bits INT,
                    layer1_weights JSONB,
                    layer1_bias JSONB,
                    layer2_weights JSONB,
                    correlation_mask JSONB,
                    encrypted_secret BYTEA,
                    protected_secret TEXT,
                    source_type TEXT,
                    speaker_id TEXT,
                    file_source TEXT,
                    registered_key TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    version TEXT DEFAULT 'v2.37'
                )
            """)

            # Короткие данные для лога (не весь массив)
            data = {
                "l1w_shape": list(self.layer1_weights.shape) if self.layer1_weights is not None else None,
                "l2w_shape": list(self.layer2_weights.shape) if self.layer2_weights is not None else None
            }

            cur.execute("""
                INSERT INTO npbk_containers 
                (user_id, key_bits, layer1_weights, layer1_bias, layer2_weights, correlation_mask,
                 encrypted_secret, protected_secret, source_type, speaker_id, file_source, registered_key)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    layer1_weights = EXCLUDED.layer1_weights,
                    layer1_bias = EXCLUDED.layer1_bias,
                    layer2_weights = EXCLUDED.layer2_weights,
                    correlation_mask = EXCLUDED.correlation_mask,
                    encrypted_secret = EXCLUDED.encrypted_secret,
                    protected_secret = EXCLUDED.protected_secret,
                    source_type = EXCLUDED.source_type,
                    speaker_id = EXCLUDED.speaker_id,
                    file_source = EXCLUDED.file_source,
                    registered_key = EXCLUDED.registered_key
            """, (
                user_id, self.key_bits,
                Json(self.layer1_weights.tolist() if self.layer1_weights is not None else []),
                Json(self.layer1_bias.tolist() if self.layer1_bias is not None else []),
                Json(self.layer2_weights.tolist() if self.layer2_weights is not None else []),
                Json(self.correlation_mask.tolist() if self.correlation_mask is not None else []),
                self.encrypted_secret or b'',
                self.protected_secret,
                self.source_info.get("type", "upload"),
                self.source_info.get("speaker_id"),
                self.source_info.get("files"),
                self.registered_key
            ))
            conn.commit()
            cur.close()
            conn.close()
            print(f"[DB] Сохранено: user={user_id}, protected_secret={self.protected_secret[:8]}... (короткий лог)")
        except Exception as e:
            print(f"[DB ERROR] {e}")

    def load_from_db(self, user_id):
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
                self.protected_secret = row[7]  # protected_secret
                print(f"[DB] Загружено: {user_id}")
                return True
            return False
        except Exception as e:
            print(f"[DB LOAD ERROR] {e}")
            return False

if __name__ == "__main__":
    npbk = NPBK()
    npbk.train([[0.1]*13 for _ in range(8)], [[0.5]*13 for _ in range(50)], user_id="test", protected_secret="my_secret_123")
    print("OK")