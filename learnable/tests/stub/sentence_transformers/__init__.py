"""Offline stand-in for sentence_transformers used only by the smoke test:
bag-of-words hash embeddings (384-d), so similar strings get similar vectors."""
import hashlib, numpy as np
class SentenceTransformer:
    def __init__(self, *a, **k): self._c = {}
    def get_sentence_embedding_dimension(self): return 384
    def _w(self, w):
        if w not in self._c:
            seed = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            self._c[w] = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        return self._c[w]
    def encode(self, text, **k):
        if isinstance(text, (list, tuple)):
            return np.stack([self.encode(t) for t in text])
        ws = str(text).lower().split() or ["unk"]
        v = sum(self._w(w) for w in ws); return (v / np.linalg.norm(v)).astype(np.float32)
