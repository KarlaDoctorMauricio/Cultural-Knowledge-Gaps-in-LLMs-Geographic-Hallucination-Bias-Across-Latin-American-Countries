"""Semantic similarity scoring for open-ended LLM answers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class SemanticSimilarityScorer:
    """Score predictions against references using multilingual sentence embeddings."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.MODEL_NAME
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, predictions: Sequence[str], references: Sequence[str]) -> np.ndarray:
        """Return cosine similarities clipped to [0, 1] for each pair."""
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        if len(predictions) == 0:
            return np.array([], dtype=float)

        pred_emb = self.model.encode(
            list(predictions),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        ref_emb = self.model.encode(
            list(references),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = np.sum(pred_emb * ref_emb, axis=1)
        return np.clip(similarities, 0.0, 1.0)

    def score_pairs(self, pairs: Iterable[tuple[str, str]]) -> np.ndarray:
        predictions, references = zip(*pairs)
        return self.score(list(predictions), list(references))
