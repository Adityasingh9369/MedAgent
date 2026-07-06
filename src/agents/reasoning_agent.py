from openai import OpenAI
from loguru import logger
from src.config import settings
from src.agents.state import AgentState
from src.ml.risk_tool import ReadmissionRiskTool, extract_patient_profile_from_query

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)

# MODEL = "google/gemini-2.0-flash-exp:free"
# MODEL = "meta-llama/llama-3.1-8b-instruct:free"
# MODEL = "google/gemma-2-9b-it:free"
MODEL = "openrouter/free"

# Load risk model once at module level
risk_tool = ReadmissionRiskTool()

SYSTEM_PROMPT = """You are a clinical decision support AI assistant.
Your job is to answer medical questions based ONLY on the provided research context.

Rules:
- Base your answer strictly on the provided context chunks
- Cite sources by mentioning the document ID
- If the context does not contain enough information, say so clearly
- Be precise and clinical in your language
- Structure your answer clearly with key points
- Never make up information not present in the context
- If a patient risk assessment is provided, incorporate it into your answer
"""


def reasoning_agent(state: AgentState) -> AgentState:
    """
    Reasoning Agent — synthesizes retrieved chunks into a cited answer.
    Also calls the ML risk tool if patient profile info is detected
    in the question.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]
    entities = state["entities"]

    logger.info("Reasoning Agent | generating answer from context...")

    # Build context string from retrieved chunks
    context = ""
    citations = []
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i} | {chunk['doc_id']} | score: {chunk['score']}]\n"
        context += chunk["content"] + "\n"
        citations.append(chunk["doc_id"])

    # Build entity summary
    entity_summary = ""
    if entities.get("diseases"):
        entity_summary += f"Identified conditions: {', '.join(entities['diseases'])}\n"
    if entities.get("drugs"):
        entity_summary += f"Identified drugs: {', '.join(entities['drugs'])}\n"
    if entities.get("symptoms"):
        entity_summary += f"Identified symptoms: {', '.join(entities['symptoms'])}\n"

    # Risk tool — call if patient profile detected in question
    risk_section = ""
    if risk_tool.is_available():
        patient_profile = extract_patient_profile_from_query(question)
        if patient_profile:
            logger.info(f"Reasoning Agent | patient profile detected: {patient_profile}")
            risk_result = risk_tool.predict(patient_profile)
            risk_section = f"""
Patient Risk Assessment (ML Model):
  Risk Score:  {risk_result['risk_score']}
  Risk Level:  {risk_result['risk_level']}
  Assessment:  {risk_result['interpretation']}
"""
            logger.info(f"Reasoning Agent | risk level: {risk_result['risk_level']}")
        else:
            logger.info("Reasoning Agent | no patient profile detected, skipping risk tool")
    else:
        logger.info("Reasoning Agent | risk model not available, skipping")

    user_prompt = f"""Clinical Question: {question}

{entity_summary}
{risk_section}
Research Context:
{context}

Please provide a comprehensive clinical answer based strictly on the above context.
Reference the source numbers when citing evidence.
If a patient risk assessment is included above, incorporate it into your recommendations.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content
    logger.info("Reasoning Agent | answer generated successfully")

    return {**state, "answer": answer, "citations": citations}