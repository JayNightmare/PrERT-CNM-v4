"""Embedding backend for the Contextual Neural Memory index.

Default backend: sentence-transformers/all-MiniLM-L6-v2 (384-dim).

If sentence-transformers is not installed, falls back to mean-pooled
all-MiniLM-L6-v2 loaded via transformers directly. Deterministic given a
fixed seed and deterministic torch state.
"""

from __future__ import annotations

import importlib
import logging
from typing import Iterable, List, Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM_DEFAULT = 384


class _SentenceTransformerEncoder:
    """Preferred backend."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_embedding_dimension() or EMBEDDING_DIM_DEFAULT)
        if self.model.get_embedding_dimension() is None:
            _LOGGER.warning(
                "----- SentenceTransformer.get_embedding_dimension() returned None; "
                "falling back to default dim=%d -----",
                EMBEDDING_DIM_DEFAULT,
            )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalise: bool = True,
    ) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalise,
        )
        return vectors.astype(np.float32)


class _TransformersMeanPoolEncoder:
    """Fallback backend using transformers + mean pooling.

    Used when sentence-transformers isn't installed. Produces the same
    embedding geometry as sentence-transformers/all-MiniLM-L6-v2 (mean-pooled
    over token embeddings with attention masking + L2 normalisation).
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        torch = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
        self._torch = torch
        self.model_name = model_name
        self.tokenizer = transformers_module.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers_module.AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        # Discover embedding dim from a probe forward pass.
        with torch.no_grad():
            probe = self.tokenizer(
                ["probe"], padding=True, truncation=True, return_tensors="pt"
            )
            probe = {k: v.to(device) for k, v in probe.items()}
            out = self.model(**probe)
            self.dim = int(out.last_hidden_state.shape[-1])

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalise: bool = True,
    ) -> np.ndarray:
        torch = self._torch
        all_vectors: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                token_embeds = out.last_hidden_state  # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
                summed = (token_embeds * mask).sum(dim=1)  # (B, H)
                counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
                pooled = summed / counts  # (B, H)
                if normalise:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(all_vectors)


class TextEncoder:
    """Facade that picks the best available backend."""

    def __init__(
        self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None
    ) -> None:
        self.model_name = model_name
        try:
            self._impl = _SentenceTransformerEncoder(model_name, device=device)
            self.backend = "sentence-transformers"
        except ModuleNotFoundError:
            _LOGGER.info(
                "sentence-transformers not installed; falling back to transformers mean-pool."
            )
            self._impl = _TransformersMeanPoolEncoder(model_name, device=device)
            self.backend = "transformers-meanpool"
        self.dim = self._impl.dim

    def encode(
        self,
        texts: Sequence[str] | Iterable[str],
        batch_size: int = 64,
        normalise: bool = True,
    ) -> np.ndarray:
        materialised = list(texts) if not isinstance(texts, list) else texts
        if not materialised:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._impl.encode(
            materialised, batch_size=batch_size, normalise=normalise
        )
