import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.services.rag_service import rag_service

def run_generation_benchmark():
    questions_file = os.path.join(os.path.dirname(__file__), "questions.json")
    if not os.path.exists(questions_file):
        print(f"Error: {questions_file} not found.")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print("=" * 60)
    print("DOCUMIND AI - RAG GENERATION & FAITHFULNESS BENCHMARK")
    print("=" * 60)

    total = len(eval_items)
    grounded_count = 0
    keyword_coverage_scores = []

    for item in eval_items:
        qid = item["id"]
        query = item["question"]
        expected_kws = item["expected_keywords"]

        res = rag_service.execute_rag_pipeline(query=query, top_k=5)
        answer = res["answer"]
        sources = res["sources"]

        # Check keyword presence in answer
        ans_lower = answer.lower()
        matched = sum(1 for kw in expected_kws if kw.lower() in ans_lower)
        coverage = (matched / len(expected_kws)) * 100
        keyword_coverage_scores.append(coverage)

        is_grounded = len(sources) > 0 and "I couldn't find enough information" not in answer
        if is_grounded:
            grounded_count += 1

        print(f"[{qid}] Grounded: {'YES' if is_grounded else 'NO'} | Keyword Coverage: {coverage:.0f}% | Sources: {len(sources)}")
        print(f"  Q: {query}")
        print(f"  A: {answer[:140]}...\n")

    avg_coverage = sum(keyword_coverage_scores) / total
    faithfulness_pct = (grounded_count / total) * 100

    print("-" * 60)
    print(f"Faithfulness & Grounded Rate: {faithfulness_pct:.2f}%")
    print(f"Average Keyword Coverage     : {avg_coverage:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_generation_benchmark()
