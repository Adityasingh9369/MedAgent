"""
ClinicalAgent — Phase 1: Data Ingestion
Entry point for running the ingestion pipeline.

Usage:
    python main.py                          # Full pipeline
    python main.py --mode pubmed            # PubMed only
    python main.py --mode pdf               # PDFs only
    python main.py --mode pubmed --queries "sepsis treatment" "diabetes care"
    python main.py --mode pubmed --max-results 100
"""
import argparse
from pathlib import Path
from loguru import logger
from src.pipeline import IngestionPipeline, DEFAULT_QUERIES


def setup_logger():
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/ingestion_{time}.log",
        rotation="100 MB",
        retention="7 days",
        level="INFO",
    )


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="ClinicalAgent Ingestion Pipeline")
    parser.add_argument(
        "--mode", choices=["pubmed", "pdf", "full"],
        default="full", help="Which ingestion to run (default: full)"
    )
    parser.add_argument(
        "--queries", nargs="+", default=None,
        help="Custom PubMed search queries"
    )
    parser.add_argument(
        "--max-results", type=int, default=50,
        help="Max articles per query (default: 50)"
    )

    args = parser.parse_args()
    pipeline = IngestionPipeline()

    if args.mode == "pubmed":
        pipeline.run_pubmed(
            queries=args.queries or DEFAULT_QUERIES,
            max_results=args.max_results
        )
    elif args.mode == "pdf":
        pipeline.run_pdf()
    else:
        pipeline.run(queries=args.queries)


if __name__ == "__main__":
    main()
