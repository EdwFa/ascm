"""Embedders. 'hash' / 'hash4' are offline, deterministic debugging stand-ins (NOT for reported results).
Real runs: BGE-M3 for the chunker, a different model (multilingual-e5-large) for the SCS metric."""
from __future__ import annotations
import hashlib, os, re, zlib
import numpy as np


class HashEmbedder:
    def __init__(self, n: int = 3, dim: int = 512):
        self.n, self.dim, self.name = n, dim, f"hash{n}"

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            t = re.sub(r'\s+', ' ', t.lower())
            for i in range(len(t) - self.n + 1):
                h = zlib.crc32(t[i:i + self.n].encode('utf-8'))
                out[r, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            out[r] = np.sign(out[r]) * np.sqrt(np.abs(out[r]))
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(nrm, 1e-9)


class STEmbedder:
    """sentence-transformers wrapper: BAAI/bge-m3, intfloat/multilingual-e5-large, ..."""
    def __init__(self, model: str, prefix: str = "", device: str | None = None, batch: int = 32):
        from sentence_transformers import SentenceTransformer
        self.name, self.prefix, self.batch = model, prefix, batch
        self.model = SentenceTransformer(model, device=device)

    def encode(self, texts):
        v = self.model.encode([self.prefix + t for t in texts], batch_size=self.batch,
                              normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def get_embedder(name: str):
    if name == "hash":
        return HashEmbedder(3)
    if name == "hash4":
        return HashEmbedder(4)
    if name in ("bge-m3", "BAAI/bge-m3"):
        return STEmbedder("BAAI/bge-m3")
    if name in ("e5", "multilingual-e5-large"):
        return STEmbedder("intfloat/multilingual-e5-large", prefix="passage: ")
    return STEmbedder(name)


def cached_encode(emb, texts, cache_dir: str | None):
    if cache_dir is None:
        return emb.encode(texts)
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1((emb.name + "\x00" + "\x01".join(texts)).encode('utf-8')).hexdigest()[:16]
    p = os.path.join(cache_dir, f"{emb.name.replace('/', '_')}_{h}.npy")
    if os.path.exists(p):
        return np.load(p)
    v = emb.encode(texts)
    np.save(p, v)
    return v


def unit_mean(vecs: np.ndarray) -> np.ndarray:
    m = vecs.mean(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-9 else m


def center(E: np.ndarray) -> np.ndarray:
    """Subtract the document mean and renormalise. Guideline sentences share a large common component, which makes raw
    cosines uniformly high; centring restores the contrast between weak and strong links."""
    C = E - E.mean(axis=0, keepdims=True)
    n = np.linalg.norm(C, axis=1, keepdims=True)
    return C / np.maximum(n, 1e-9)
