"""
End-to-end test for the LangGraph multi-agent pipeline.
Run with:
    python -m scripts.test_pipeline
"""
from src.agents.pipeline import run_pipeline


def test(question: str):
    result = run_pipeline(question)
    print(result["final_response"])
    print("=" * 55)


if __name__ == "__main__":
    questions = [
        "What are the treatment options for heart failure with reduced ejection fraction?",
        "What antibiotics are recommended for community acquired pneumonia?",
        "What is the prognosis for a 72-year-old male with heart failure and renal disease after a 7-day hospital stay?",
]

    for q in questions:
        test(q)