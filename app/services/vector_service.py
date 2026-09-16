import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import faiss
import numpy as np
from app.core.config import settings
from app.services.embedding_service import embedding_service

logger = logging.getLogger("documind.vector")

class VectorSearchResult:
    def __init__(self, chunk_id: str, document_id: str, document_name: str, page_number: int, chunk_index: int, text: str, similarity_score: float):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_name = document_name
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.text = text
        self.similarity_score = float(similarity_score)

class FAISSVectorService:
    def __init__(self, store_dir: str = None):
        self.store_dir = store_dir or settings.VECTOR_STORE_DIR
        self.index_path = os.path.join(self.store_dir, "faiss.index")
        self.metadata_path = os.path.join(self.store_dir, "metadata.json")
        
        self.dim = 384  # default all-MiniLM-L6-v2 dimension
        self.index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}  # faiss_id -> chunk_meta
        self.vector_count = 0
        
        self._load_or_create_index()

    def _load_or_create_index(self):
        os.makedirs(self.store_dir, exist_ok=True)
        self.dim = embedding_service.embedding_dimension
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                    self.metadata = {int(k): v for k, v in raw_meta.items()}
                self.vector_count = self.index.ntotal
                logger.info(f"Loaded existing FAISS index with {self.vector_count} vectors.")
                return
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Initializing fresh index.")

        # Create Flat Inner Product index (equivalent to Cosine Similarity for normalized vectors)
        base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base_index)
        self.metadata = {}
        self.vector_count = 0
        self.save_index()

    def save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Successfully saved FAISS index with {self.index.ntotal} vectors to disk.")
        except Exception as e:
            logger.error(f"Error saving FAISS index to disk: {e}")

    def add_chunks(self, chunks_data: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        if not chunks_data or not embeddings:
            return []

        vectors_np = np.array(embeddings, dtype=np.float32)
        next_id = max(self.metadata.keys(), default=-1) + 1
        faiss_ids = np.arange(next_id, next_id + len(chunks_data), dtype=np.int64)

        vector_ids = []
        for i, chunk_meta in enumerate(chunks_data):
            fid = int(faiss_ids[i])
            vector_id = f"vec_{fid}"
            chunk_meta["faiss_id"] = fid
            chunk_meta["vector_id"] = vector_id
            self.metadata[fid] = chunk_meta
            vector_ids.append(vector_id)

        self.index.add_with_ids(vectors_np, faiss_ids)
        self.vector_count = self.index.ntotal
        self.save_index()
        return vector_ids

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[VectorSearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []

        # Over-fetch if filtering by document_ids
        fetch_k = top_k * 5 if document_ids else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        q_arr = np.array([query_vector], dtype=np.float32)
        scores, ids = self.index.search(q_arr, fetch_k)

        results: List[VectorSearchResult] = []
        doc_set = set(document_ids) if document_ids else None

        for score, fid in zip(scores[0], ids[0]):
            if fid == -1 or fid not in self.metadata:
                continue
            
            meta = self.metadata[fid]
            doc_id = meta.get("document_id")
            
            if doc_set and doc_id not in doc_set:
                continue

            # Normalized score range [0.0, 1.0]
            sim_score = max(0.0, min(1.0, float(score)))

            results.append(VectorSearchResult(
                chunk_id=meta.get("chunk_id", ""),
                document_id=doc_id,
                document_name=meta.get("document_name", "Unknown Document"),
                page_number=meta.get("page_number", 1),
                chunk_index=meta.get("chunk_index", 0),
                text=meta.get("text", ""),
                similarity_score=sim_score
            ))

            if len(results) >= top_k:
                break

        return results

    def delete_document_chunks(self, document_id: str):
        if self.index is None or self.index.ntotal == 0:
            return

        ids_to_remove = [fid for fid, meta in self.metadata.items() if meta.get("document_id") == document_id]
        if not ids_to_remove:
            return

        arr_to_remove = np.array(ids_to_remove, dtype=np.int64)
        self.index.remove_ids(arr_to_remove)

        for fid in ids_to_remove:
            self.metadata.pop(fid, None)

        self.vector_count = self.index.ntotal
        self.save_index()
        logger.info(f"Deleted {len(ids_to_remove)} vectors for document {document_id}")

vector_service = FAISSVectorService()
