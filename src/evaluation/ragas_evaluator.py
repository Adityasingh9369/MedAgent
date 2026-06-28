# src/evaluation/ragas_evaluator.py

import os
from loguru import logger
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import settings

# Set LangSmith env vars
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT


def get_ragas_llm():
    """Returns a LangchainLLMWrapper pointing to OpenRouter."""
    llm = ChatOpenAI(
        model="openrouter/auto",
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    """Returns embeddings wrapper pointing to OpenRouter."""
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas_evaluation(pipeline_results: list[dict]) -> dict:
    """
    Takes a list of pipeline results and runs RAGAS evaluation.

    Each item in pipeline_results must have:
      - question: str
      - answer: str
      - retrieved_chunks: list of dicts with 'content' key
      - ground_truth: str

    Returns a dict of metric scores.
    """
    logger.info("Preparing RAGAS dataset...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for result in pipeline_results:
        questions.append(result["question"])
        answers.append(result["answer"])
        contexts.append([chunk["content"] for chunk in result["retrieved_chunks"]])
        ground_truths.append(result["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    logger.info(f"Running RAGAS on {len(questions)} samples...")

    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Attach our LLM and embeddings to each metric
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    scores = evaluate(
        dataset,
        metrics=metrics,
    )

    return scores