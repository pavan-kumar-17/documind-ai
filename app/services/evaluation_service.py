import json
import time
from datetime import datetime
from typing import List, Dict, Any
from app.services.retrieval_service import retrieval_service
from app.schemas.pydantic_schemas import EvaluationRunResponse

class EvaluationService:
    @classmethod
    def run_retrieval_evaluation(cls, top_k: int = 5) -> EvaluationRunResponse:
        sample_eval_set = [
            {
                "question": "What is the primary revenue growth rate?",
                "expected_keywords": ["revenue", "growth", "percent", "financial"]
            },
            {
                "question": "What are the company's ESG and sustainability goals?",
                "expected_keywords": ["sustainability", "esg", "carbon", "emission", "green"]
            },
            {
                "question": "What are the risk factors associated with cybersecurity?",
                "expected_keywords": ["risk", "cybersecurity", "security", "threat", "data"]
            }
        ]

        total_questions = len(sample_eval_set)
        hits = 0
        reciprocal_ranks = []
        precision_scores = []
        latencies = []

        details = []

        for item in sample_eval_set:
            q = item["question"]
            keywords = item["expected_keywords"]

            t0 = time.time()
            results = retrieval_service.retrieve(query=q, top_k=top_k)
            t_elapsed = (time.time() - t0) * 1000
            latencies.append(t_elapsed)

            matched_count = 0
            first_rank = 0

            for rank, res in enumerate(results, 1):
                text_lower = res.text.lower()
                if any(kw in text_lower for kw in keywords):
                    matched_count += 1
                    if first_rank == 0:
                        first_rank = rank

            is_hit = matched_count > 0
            if is_hit:
                hits += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

            precision_scores.append(matched_count / max(1, len(results)))

            details.append({
                "question": q,
                "hit": is_hit,
                "first_rank": first_rank,
                "retrieved_count": len(results),
                "precision": round(matched_count / max(1, len(results)), 2),
                "latency_ms": round(t_elapsed, 2)
            })

        hit_rate = hits / max(1, total_questions)
        mrr = sum(reciprocal_ranks) / max(1, total_questions)
        avg_precision = sum(precision_scores) / max(1, total_questions)
        avg_latency = sum(latencies) / max(1, total_questions)

        return EvaluationRunResponse(
            total_questions=total_questions,
            retrieval_hit_rate=round(hit_rate, 4),
            retrieval_mrr=round(mrr, 4),
            retrieval_precision_at_k=round(avg_precision, 4),
            avg_response_time_ms=round(avg_latency, 2),
            evaluated_at=datetime.utcnow(),
            details=details
        )
