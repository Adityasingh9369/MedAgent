"""
Exports the Phase 3 eval questions with their retrieved contexts
to a JSON file, for use in Colab base-vs-finetuned comparison.

Run with:
    python -m scripts.export_eval_context
"""
import json
from loguru import logger
from src.agents.retrieval_agent import retrieval_agent
from src.evaluation.eval_dataset import EVAL_DATASET

def main():
    export_data = []

    for item in EVAL_DATASET:
        state = {"question": item["question"]}
        result = retrieval_agent(state)

        export_data.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "retrieved_chunks": [
                {"content": c["content"], "doc_id": c["doc_id"]}
                for c in result["retrieved_chunks"]
            ],
        })

        logger.info(f"Exported: {item['question'][:60]}...")

    output_path = "phase5_eval_context.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"Saved {len(export_data)} questions to {output_path}")

if __name__ == "__main__":
    main()