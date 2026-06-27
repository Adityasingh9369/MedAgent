from openai import OpenAI
from loguru import logger
import re
from src.config import settings
from src.agents.state import AgentState


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)

# MODEL = "google/gemini-2.0-flash-exp:free"
# MODEL = "meta-llama/llama-3.1-8b-instruct:free"
# MODEL = "google/gemma-2-9b-it:free"
MODEL = "openrouter/free"

CRITIQUE_PROMPT = """You are a medical fact-checker reviewing an AI-generated clinical answer.

Your job:
1. Check if the answer is grounded in the provided context
2. Identify any claims not supported by the context
3. Flag potential hallucinations
4. Assign a confidence score from 0.0 to 1.0

Scoring guide:
  0.9 - 1.0 : Answer fully grounded, all claims supported by context
  0.7 - 0.9 : Mostly grounded, minor gaps
  0.5 - 0.7 : Partially grounded, some unsupported claims
  0.0 - 0.5 : Significant hallucination risk, do not trust

Respond in this exact format:
CRITIQUE: <your critique here>
CONFIDENCE: <score between 0.0 and 1.0>
RELIABLE: <YES or NO>
"""


def parse_critique_response(response_text: str) -> tuple:
    critique = ""
    confidence = 0.5
    reliable = False

    lines = response_text.strip().split("\n")
    for line in lines:
        if line.startswith("CRITIQUE:"):
            critique = line.replace("CRITIQUE:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                match = re.search(r"[\d.]+", line.replace("CONFIDENCE:", ""))
                if match:
                    confidence = float(match.group())
                    confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        elif line.startswith("RELIABLE:"):
            reliable = line.replace("RELIABLE:", "").strip().upper() == "YES"

    return critique, confidence, reliable


def critique_agent(state: AgentState) -> AgentState:
    """
    Critique Agent — fact-checks the Reasoning Agent's answer.
    Compares generated answer against source context,
    assigns confidence score, and flags hallucinations.
    """
    question = state["question"]
    answer = state["answer"]
    chunks = state["retrieved_chunks"]

    logger.info("Critique Agent | fact-checking the generated answer...")

    context = "\n".join([
        f"[Source {i+1}]: {chunk['content'][:300]}"
        for i, chunk in enumerate(chunks)
    ])

    user_prompt = f"""Original Question: {question}

Source Context Used:
{context}

Generated Answer to Review:
{answer}

Please critique this answer and provide your assessment.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CRITIQUE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    critique, confidence, reliable = parse_critique_response(
        response.choices[0].message.content
    )

    logger.info(f"Critique Agent | confidence: {confidence} | reliable: {reliable}")

    reliability_label = "✓ RELIABLE" if reliable else "⚠ LOW CONFIDENCE"
    final_response = f"""
{reliability_label} (Score: {confidence:.2f})

ANSWER:
{answer}

SOURCES:
{chr(10).join([f"  - {c}" for c in state['citations']])}

CRITIQUE:
{critique}
"""

    return {
        **state,
        "critique": critique,
        "confidence_score": confidence,
        "is_reliable": reliable,
        "final_response": final_response,
    }