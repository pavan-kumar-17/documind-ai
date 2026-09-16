import json
import logging
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger("documind.llm")

GROUNDED_SYSTEM_PROMPT = """You are DocuMind AI, an enterprise-grade document intelligence assistant.
Your goal is to provide accurate, concise, and professional answers strictly grounded in the provided document context.

RULES FOR ANSWER GENERATION:
1. Use ONLY the facts provided in the DOCUMENT CONTEXT below to answer the user question.
2. Do NOT invent, assume, or extrapolate facts outside the context.
3. If the retrieved document context does NOT contain sufficient details to answer the query, state explicitly:
   "I couldn't find enough information in the uploaded documents to answer this reliably."
4. When answering, cite the sources implicitly or reference the key findings.
5. Keep your answer clear, structured, and professional.

DOCUMENT CONTEXT:
{context_text}
"""

class LLMService:
    def generate_grounded_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "I couldn't find enough information in the uploaded documents to answer this reliably."

        # Format context block
        context_blocks = []
        for i, chunk in enumerate(context_chunks, 1):
            doc_name = chunk.get("document_name", "Document")
            page_num = chunk.get("page_number", 1)
            text = chunk.get("text", "")
            context_blocks.append(f"[{i}] {doc_name} (Page {page_num}):\n{text}")

        formatted_context = "\n\n".join(context_blocks)
        system_prompt = GROUNDED_SYSTEM_PROMPT.format(context_text=formatted_context)

        provider = settings.LLM_PROVIDER.lower()

        try:
            if provider == "openai" and settings.OPENAI_API_KEY:
                return self._call_openai(system_prompt, query)
            elif provider == "gemini" and settings.GEMINI_API_KEY:
                return self._call_gemini(system_prompt, query)
            elif provider == "ollama":
                return self._call_ollama(system_prompt, query)
            else:
                # Default smart contextual mock engine if no key provided
                return self._call_smart_mock(query, context_chunks)
        except Exception as e:
            logger.error(f"Error calling LLM provider '{provider}': {e}. Falling back to Smart Context Extractor.")
            return self._call_smart_mock(query, context_chunks)

    def _call_openai(self, system_prompt: str, user_query: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.2
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, system_prompt: str, user_query: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\nUser Question: {user_query}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_ollama(self, system_prompt: str, user_query: str) -> str:
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\nUser Question: {user_query}",
            "stream": False,
            "options": {"temperature": 0.2}
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["response"].strip()

    def _call_smart_mock(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Deterministic, offline smart context extraction engine.
        Synthesizes the retrieved chunks grounded in exact document text.
        """
        query_words = set(query.lower().split())
        matched_sentences = []

        for chunk in context_chunks:
            text = chunk.get("text", "")
            doc_name = chunk.get("document_name", "Document")
            page_num = chunk.get("page_number", 1)
            
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            for sentence in sentences:
                s_words = set(sentence.lower().split())
                overlap = query_words.intersection(s_words)
                if len(overlap) >= 2 or any(w in sentence.lower() for w in query_words if len(w) > 4):
                    matched_sentences.append((sentence, doc_name, page_num))

        if not matched_sentences:
            # Fallback to top retrieved chunk text
            first_chunk = context_chunks[0]
            doc_name = first_chunk.get("document_name", "the uploaded document")
            page_num = first_chunk.get("page_number", 1)
            snippet = first_chunk.get("text", "")[:350]
            return f"Based on {doc_name} (Page {page_num}):\n\n\"{snippet}...\"\n\n(Note: Generated via DocuMind Grounded Intelligence Service)"

        # Deduplicate & construct response
        unique_matches = []
        seen = set()
        for sent, doc, pg in matched_sentences:
            if sent not in seen:
                seen.add(sent)
                unique_matches.append((sent, doc, pg))

        response_lines = ["Based on the retrieved document context:\n"]
        for sent, doc, pg in unique_matches[:4]:
            response_lines.append(f"• **{doc} (Page {pg})**: {sent}.")

        return "\n".join(response_lines)

llm_service = LLMService()
