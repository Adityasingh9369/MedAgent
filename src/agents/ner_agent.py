import spacy
from loguru import logger
from src.agents.state import AgentState


nlp = spacy.load("en_core_web_sm")

# Medical keyword lists for pattern matching
# spaCy's general model doesn't know medical terms
# so we combine it with simple keyword matching
DISEASE_KEYWORDS = [
    "diabetes", "hypertension", "pneumonia", "sepsis", "heart failure",
    "myocardial infarction", "stroke", "kidney disease", "asthma", "copd",
    "cancer", "hepatitis", "tuberculosis", "hiv", "covid", "dementia",
    "alzheimer", "parkinson", "arthritis", "osteoporosis", "anemia",
]

DRUG_KEYWORDS = [
    "metformin", "insulin", "aspirin", "warfarin", "heparin", "amoxicillin",
    "penicillin", "azithromycin", "lisinopril", "atorvastatin", "metoprolol",
    "furosemide", "prednisone", "dexamethasone", "vancomycin", "ceftriaxone",
    "ciprofloxacin", "omeprazole", "amlodipine", "losartan", "ramipril",
]

SYMPTOM_KEYWORDS = [
    "fever", "cough", "dyspnea", "chest pain", "fatigue", "nausea",
    "vomiting", "diarrhea", "headache", "dizziness", "edema", "hypotension",
    "hypertension", "tachycardia", "bradycardia", "syncope", "seizure",
    "confusion", "weakness", "pain", "shortness of breath", "palpitations",
]


def extract_entities(text: str) -> dict:
    """Extract medical entities using spaCy + keyword matching."""
    text_lower = text.lower()
    doc = nlp(text)

    # spaCy named entities (catches proper nouns, organizations etc.)
    spacy_entities = [ent.text for ent in doc.ents]

    # Keyword matching for medical terms
    diseases = [kw for kw in DISEASE_KEYWORDS if kw in text_lower]
    drugs = [kw for kw in DRUG_KEYWORDS if kw in text_lower]
    symptoms = [kw for kw in SYMPTOM_KEYWORDS if kw in text_lower]

    return {
        "diseases": list(set(diseases)),
        "drugs": list(set(drugs)),
        "symptoms": list(set(symptoms)),
        "spacy_entities": spacy_entities,
    }


def ner_agent(state: AgentState) -> AgentState:
    """
    Medical NER Agent — extracts clinical entities from the question.
    Identifies diseases, drugs, and symptoms mentioned.
    This enriches the context passed to the Reasoning Agent.
    """
    question = state["question"]
    logger.info(f"NER Agent | extracting entities from question...")

    entities = extract_entities(question)

    logger.info(f"NER Agent | diseases: {entities['diseases']}")
    logger.info(f"NER Agent | drugs: {entities['drugs']}")
    logger.info(f"NER Agent | symptoms: {entities['symptoms']}")

    return {**state, "entities": entities}