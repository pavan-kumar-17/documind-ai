import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger("documind.embedding")

class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        if not text.strip():
            # Return zero vector if empty string
            dimension = self.model.get_sentence_embedding_dimension()
            return [0.0] * dimension
            
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
            
        clean_texts = [t if t.strip() else " " for t in texts]
        vectors = self.model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vectors.tolist()

    @property
    def embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

embedding_service = EmbeddingService()
