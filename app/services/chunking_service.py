import re
from typing import List, Dict, Any
from app.core.config import settings
from app.services.document_service import ExtractedDocument

class TextChunk:
    def __init__(self, page_number: int, chunk_index: int, text: str):
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.text = text
        self.char_count = len(text)
        self.token_count = len(text.split())  # Word count approximation for tokens

class ChunkingService:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

    def chunk_extracted_document(self, extracted_doc: ExtractedDocument) -> List[TextChunk]:
        all_chunks: List[TextChunk] = []
        global_chunk_index = 0

        for page_num, page_text in extracted_doc.pages:
            page_chunks = self._chunk_text(page_text, page_num, global_chunk_index)
            for chunk in page_chunks:
                all_chunks.append(chunk)
                global_chunk_index += 1

        return all_chunks

    def _chunk_text(self, text: str, page_number: int, start_index: int) -> List[TextChunk]:
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', text).strip()
        if not clean_text:
            return []

        # Split into sentence units
        sentences = re.split(r'(?<=[.!?]) +', clean_text)
        chunks: List[TextChunk] = []
        
        current_chunk_words: List[str] = []
        current_length = 0
        chunk_idx = start_index

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            if current_length + sentence_word_count > (self.chunk_size // 5):  # Convert char length config to ~words
                if current_chunk_words:
                    chunk_str = " ".join(current_chunk_words)
                    chunks.append(TextChunk(page_number, chunk_idx, chunk_str))
                    chunk_idx += 1

                    # Overlap: retain last N words
                    overlap_words_count = self.chunk_overlap // 5
                    current_chunk_words = current_chunk_words[-overlap_words_count:] if overlap_words_count > 0 else []
                    current_length = len(current_chunk_words)

            current_chunk_words.extend(sentence_words)
            current_length += sentence_word_count

        if current_chunk_words:
            chunk_str = " ".join(current_chunk_words)
            chunks.append(TextChunk(page_number, chunk_idx, chunk_str))

        return chunks
