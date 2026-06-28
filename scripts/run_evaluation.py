"""
Runs the full RAGAS evaluation pipeline.
Feeds each question through the agent pipeline,
collects results, and scores them with RAGAS.

Run with:
    python -m scripts.run_evaluation
"""

from loguru import logger
from src.agents.pipeline import run_pipeline
from src.evaluation.eval_dataset import EVAL_DATASET
from src.evaluation.ragas_evaluator import run_ragas_evaluation


def main():
    logger.info("=" * 55)
    logger.info("RAGAS EVALUATION STARTED")
    logger.info(f"Total questions: {len(EVAL_DATASET)}")
    logger.info("=" * 55)

    pipeline_results = []

    for i, item in enumerate(EVAL_DATASET, 1):
        logger.info(f"Running question {i}/{len(EVAL_DATASET)}: {item['question'][:60]}...")

        result = run_pipeline(item["question"])

        pipeline_results.append({
            "question": item["question"],
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "ground_truth": item["ground_truth"],
        })

        logger.info(f"Question {i} done | confidence: {result['confidence_score']:.2f}")

    logger.info("=" * 55)
    logger.info("All questions processed. Running RAGAS scoring...")
    logger.info("=" * 55)

    scores = run_ragas_evaluation(pipeline_results)

    print("\n" + "=" * 55)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 55)

    # changed here:-
    def to_float(val):
        return float(val[0]) if isinstance(val, list) else float(val)

    f = to_float(scores['faithfulness'])
    ar = to_float(scores['answer_relevancy'])
    cp = to_float(scores['context_precision'])
    cr = to_float(scores['context_recall'])

    print(f"  Faithfulness:       {f:.4f}")
    print(f"  Answer Relevancy:   {ar:.4f}")
    print(f"  Context Precision:  {cp:.4f}")
    print(f"  Context Recall:     {cr:.4f}")
    print("=" * 55)

    avg = (f + ar + cp + cr) / 4
    # till here
    
    print(f"  Overall Average:    {avg:.4f}")
    print("=" * 55)

    return scores


if __name__ == "__main__":
    main()