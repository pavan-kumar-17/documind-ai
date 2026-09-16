import sys
import os
import json
import time

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.services.retrieval_service import retrieval_service

def run_retrieval_benchmark():
    questions_file = os.path.join(os.path.dirname(__file__), "questions.json")
    if not os.path.exists(questions_file):
        print(f"Error: {questions_file} not found.")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print("=" * 60)
    print("DOCUMIND AI - RAG RETRIEVAL EVALUATION BENCHMARK")
    print("=" * 60)

    top_k = 5
    total = len(eval_items)
    hits = 0
    reciprocal_ranks = []
    precision_scores = []
    latencies = []

    for item in eval_items:
        qid = item["id"]
        query = item["question"]
        expected_keywords = item["expected_keywords"]

        t0 = time.time()
        results = retrieval_service.retrieve(query=query, top_k=top_k)
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        found_match = False
        first_rank = 0

        for rank, res in enumerate(results, 1):
            text_lower = res.text.lower()
            if any(kw.lower() in text_lower for kw in expected_keywords):
                found_match = True
                if first_rank == 0:
                    first_rank = rank

        if found_match:
            hits += 1
            rr = 1.0 / first_rank
            reciprocal_ranks.append(rr)
            print(f"[SUCCESS] {qid}: Hit at Rank {first_rank} ({elapsed_ms:.1f}ms)")
        else:
            reciprocal_ranks.append(0.0)
            print(f"[MISS]    {qid}: No keyword match found ({elapsed_ms:.1f}ms)")

    hit_rate = (hits / total) * 100
    mrr = (sum(reciprocal_ranks) / total)
    avg_lat = sum(latencies) / total

    print("-" * 60)
    print(f"Total Test Questions : {total}")
    print(f"Hit Rate @ K={top_k}     : {hit_rate:.2f}%")
    print(f"MRR (Mean Reciprocal): {mrr:.4f}")
    print(f"Average Latency      : {avg_lat:.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    run_retrieval_benchmark()
