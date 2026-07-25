"""
Compares base model vs fine-tuned v2 model on RAGAS metrics.
Run with:
    python -m scripts.compare_finetuning
"""
import json
from loguru import logger
from src.evaluation.ragas_evaluator import run_ragas_evaluation


def to_float(val):
    return float(val[0]) if isinstance(val, list) else float(val)


def score_and_print(label, results):
    logger.info(f"Scoring {label}...")
    scores = run_ragas_evaluation(results)

    f = to_float(scores["faithfulness"])
    ar = to_float(scores["answer_relevancy"])
    cp = to_float(scores["context_precision"])
    cr = to_float(scores["context_recall"])
    avg = (f + ar + cp + cr) / 4

    print(f"\n{'=' * 55}")
    print(f"{label}")
    print(f"{'=' * 55}")
    print(f"  Faithfulness:       {f:.4f}")
    print(f"  Answer Relevancy:   {ar:.4f}")
    print(f"  Context Precision:  {cp:.4f}")
    print(f"  Context Recall:     {cr:.4f}")
    print(f"  Overall Average:    {avg:.4f}")
    print(f"{'=' * 55}")

    return {"faithfulness": f, "answer_relevancy": ar,
            "context_precision": cp, "context_recall": cr, "avg": avg}


def main():
    with open("base_results.json", "r", encoding="utf-8") as f:
        base_results = json.load(f)
    with open("finetuned_v2_results.json", "r", encoding="utf-8") as f:
        finetuned_results = json.load(f)

    base_scores = score_and_print("BASE MODEL (Llama-3.1-8B-Instruct)", base_results)
    ft_scores = score_and_print("FINE-TUNED v2 (LoRA, mixed-format MedQA)", finetuned_results)

    print(f"\n{'=' * 55}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 55}")
    print(f"{'Metric':<20}{'Base':<12}{'Fine-tuned v2':<15}{'Delta':<10}")
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "avg"]:
        delta = ft_scores[key] - base_scores[key]
        sign = "+" if delta >= 0 else ""
        print(f"{key:<20}{base_scores[key]:<12.4f}{ft_scores[key]:<15.4f}{sign}{delta:.4f}")
    print(f"{'=' * 55}")

    print("\nPhase 3 pipeline baseline (full agent system, for reference):")
    print("  Faithfulness: 0.79 | Answer Relevancy: 0.78 | Context Precision: 0.00 | Context Recall: 0.00")


if __name__ == "__main__":
    main()