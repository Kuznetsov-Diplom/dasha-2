#!/usr/bin/env python3
"""
NPBK v2.38.10 — quality gate (FRR/FAR check after training)

- registered_key не храним открыто
- Сразу после train() считаем ошибки 1 и 2 рода
- Если FRR > 10% или хаос у чужих мало — не сохраняем
"""

import numpy as np
import psycopg2
from psycopg2.extras import Json
import os
import base64
import hashlib
from typing import Optional, Tuple

try:
    import gostcrypto
    from gostcrypto.gostcipher import GOSTCipher
    GOSTCRYPTO_AVAILABLE = True
except ImportError:
    GOSTCRYPTO_AVAILABLE = False
    gostcrypto = None

class NPBK:
    def __init__(self, input_dim=13, key_bits=128, db_url=None, use_kuznechik=True):
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
        self.encrypted_secret = None
        self.protected_secret = None
        self.use_kuznechik = use_kuznechik and GOSTCRYPTO_AVAILABLE
        self.quality_report = {}

    def _to_bytes(self, data):
        if data is None: return b""
        if isinstance(data, (bytes, bytearray)): return bytes(data)
        if isinstance(data, memoryview): return data.tobytes()
        if isinstance(data, str): return data.encode("utf-8", errors="replace")
        try: return bytes(data)
        except: return b""

    def _kuznechik_encrypt(self, plaintext, key128):
        plaintext = self._to_bytes(plaintext)
        key128 = self._to_bytes(key128)
        if not plaintext: return b""
        if len(key128) < 16: key128 = key128.ljust(16, b"\0")
        if not self.use_kuznechik or gostcrypto is None:
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
        except:
            expanded = (key128 * (len(plaintext) // 16 + 2))[:len(plaintext)]
            ct = bytes(p ^ k for p, k in zip(plaintext, expanded))
            return base64.b64encode(ct)

    def _kuznechik_decrypt(self, ciphertext, key128):
        ciphertext = self._to_bytes(ciphertext)
        key128 = self._to_bytes(key128)
        if not ciphertext: return b""
        try: ct = base64.b64decode(ciphertext) if ciphertext else b""
        except: ct = ciphertext
        ct = self._to_bytes(ct)
        if len(key128) < 16: key128 = key128.ljust(16, b"\0")
        if not self.use_kuznechik or gostcrypto is None:
            expanded = (key128 * (len(ct) // 16 + 2))[:len(ct)]
            return bytes(c ^ k for c, k in zip(ct, expanded))
        try:
            key256 = hashlib.sha256(key128).digest()
            cipher = GOSTCipher("kuznechik", key256)
            padded = cipher.decrypt(ct)
            pad_len = padded[-1] if padded else 0
            return padded[:-pad_len] if pad_len > 0 else padded
        except:
            expanded = (key128 * (len(ct) // 16 + 2))[:len(ct)]
            return bytes(c ^ k for c, k in zip(ct, expanded))

    def train(self, own_vectors, alien_vectors, user_id="default", source_info=None, protected_secret=None) -> Tuple[bool, dict]:
        source_info = source_info or {"type": "upload"}
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
        self.trained = True
        self.user_id = user_id

        own_keys = [self.generate_key(v) for v in own[:min(8, len(own))]]
        alien_keys = [self.generate_key(v) for v in alien[:min(20, len(alien))]]

        unique_own = len(set(own_keys))
        frr = 0.0 if unique_own == 1 else (unique_own - 1) / len(own_keys)

        unique_alien = len(set(alien_keys))
        far = 1.0 - (unique_alien / len(alien_keys)) if len(alien_keys) > 0 else 1.0

        self.quality_report = {"FRR": round(frr, 4), "FAR": round(far, 4), "unique_own": unique_own, "unique_alien": unique_alien}

        if frr > 0.10 or far < 0.70:
            self.trained = False
            print(f"[NPBK] ОБУЧЕНИЕ ПРОВАЛЕНО! FRR={frr:.1%}, FAR={far:.1%}")
            return False, self.quality_report

        key_bytes = bytes(int(own_keys[0][i:i+8], 2) for i in range(0, 128, 8))
        self.encrypted_secret = self._kuznechik_encrypt(self.protected_secret.encode("utf-8"), key_bytes)
        self.save_to_db(user_id)
        print(f"[NPBK] Обучение успешно! FRR={frr:.1%}, FAR={far:.1%}")
        return True, self.quality_report

    def _compute_stats(self, v): return np.mean(v, axis=0), np.std(v, axis=0, ddof=1)
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
        if self.layer1_weights is not None: self.layer1_weights *= mask

    def generate_key(self, vec):
        if not self.trained or self.layer1_weights is None: raise ValueError("Not trained")
        y = np.array(vec) @ self.layer1_weights.T + self.layer1_bias
        bits = (y > 0).astype(int)
        if self.layer2_weights is not None: bits = (bits @ self.layer2_weights.T > 0).astype(int)
        return "".join(map(str, bits))

    def save_to_db(self, user_id):
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS npbk_containers (user_id TEXT PRIMARY KEY, key_bits INT, layer1_weights JSONB, layer1_bias JSONB, layer2_weights JSONB, correlation_mask JSONB, encrypted_secret BYTEA, source_type TEXT, created_at TIMESTAMP DEFAULT NOW(), version TEXT DEFAULT 'v2.38.10')")
            cur.execute("INSERT INTO npbk_containers (user_id, key_bits, layer1_weights, layer1_bias, layer2_weights, correlation_mask, encrypted_secret, source_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (user_id) DO UPDATE SET layer1_weights=EXCLUDED.layer1_weights, layer1_bias=EXCLUDED.layer1_bias, layer2_weights=EXCLUDED.layer2_weights, correlation_mask=EXCLUDED.correlation_mask, encrypted_secret=EXCLUDED.encrypted_secret, source_type=EXCLUDED.source_type", (user_id, self.key_bits, Json(self.layer1_weights.tolist() if self.layer1_weights is not None else []), Json(self.layer1_bias.tolist() if self.layer1_bias is not None else []), Json(self.layer2_weights.tolist() if self.layer2_weights is not None else []), Json(self.correlation_mask.tolist() if self.correlation_mask is not None else []), self.encrypted_secret or b"", self.source_info.get("type", "upload")))
            conn.commit()
            cur.close()
            conn.close()
            print(f"[DB] Saved: {user_id} (only encrypted_secret)")
        except Exception as e: print(f"[DB ERROR] {e}")

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
                self.key_bits = row[1] or 128
                self.layer1_weights = np.array(row[2]) if row[2] else None
                self.layer1_bias = np.array(row[3]) if row[3] else None
                self.layer2_weights = np.array(row[4]) if row[4] else None
                self.correlation_mask = np.array(row[5]) if row[5] else None
                self.encrypted_secret = row[6]
                self.protected_secret = None
                return True
            return False
        except Exception as e: print(f"[DB LOAD ERROR] {e}"); return False

if __name__ == "__main__":
    npbk = NPBK()
    ok, rep = npbk.train([[0.1]*13 for _ in range(8)], [[0.5]*13 for _ in range(50)], "test", protected_secret="secret123")
    print("Quality:", rep)
    print("SUCCESS" if ok else "FAILED")