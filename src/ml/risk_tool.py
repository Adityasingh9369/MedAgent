# src/ml/risk_tool.py

import numpy as np
from xgboost import XGBClassifier
from loguru import logger
from pathlib import Path

MODEL_PATH = Path("models/readmission_model.json")

FEATURE_COLS = [
    "age", "gender_m", "admission_type_emergency",
    "los_days", "diagnosis_count",
    "has_diabetes", "has_heart_failure", "has_hypertension",
    "has_renal_disease", "has_pneumonia", "has_sepsis", "has_copd",
    "num_icu_stays", "total_icu_los", "max_icu_los",
]


class ReadmissionRiskTool:
    """
    Wraps the trained XGBoost readmission model as a callable tool
    that the Reasoning Agent can invoke when patient profile
    information is present in the query.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            logger.warning(f"Risk model not found at {MODEL_PATH}. Train it first with scripts/train_risk_model.py")
            return
        self.model = XGBClassifier()
        self.model.load_model(str(MODEL_PATH))
        logger.info("Readmission risk model loaded successfully")

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, patient_profile: dict) -> dict:
        """
        Predict 30-day readmission risk from a patient profile dict.

        Expected keys (all optional, defaults to 0/mean):
            age, gender_m, admission_type_emergency, los_days,
            diagnosis_count, has_diabetes, has_heart_failure,
            has_hypertension, has_renal_disease, has_pneumonia,
            has_sepsis, has_copd, num_icu_stays,
            total_icu_los, max_icu_los

        Returns:
            dict with risk_score, risk_level, interpretation
        """
        if not self.is_available():
            return {
                "error": "Risk model not available. Run train_risk_model.py first.",
                "risk_score": None,
                "risk_level": None,
            }

        # Build feature vector with safe defaults
        features = []
        defaults = {
            "age": 65.0,
            "gender_m": 0,
            "admission_type_emergency": 0,
            "los_days": 5.0,
            "diagnosis_count": 3.0,
            "has_diabetes": 0,
            "has_heart_failure": 0,
            "has_hypertension": 0,
            "has_renal_disease": 0,
            "has_pneumonia": 0,
            "has_sepsis": 0,
            "has_copd": 0,
            "num_icu_stays": 0,
            "total_icu_los": 0.0,
            "max_icu_los": 0.0,
        }

        for col in FEATURE_COLS:
            features.append(float(patient_profile.get(col, defaults[col])))

        feature_array = np.array(features).reshape(1, -1)
        risk_score = float(self.model.predict_proba(feature_array)[0][1])

        # Risk stratification
        if risk_score >= 0.6:
            risk_level = "HIGH"
            interpretation = (
                f"High 30-day readmission risk ({risk_score:.1%}). "
                "Consider enhanced discharge planning, early follow-up appointment, "
                "and patient education on warning signs."
            )
        elif risk_score >= 0.3:
            risk_level = "MODERATE"
            interpretation = (
                f"Moderate 30-day readmission risk ({risk_score:.1%}). "
                "Standard discharge planning with scheduled follow-up recommended."
            )
        else:
            risk_level = "LOW"
            interpretation = (
                f"Low 30-day readmission risk ({risk_score:.1%}). "
                "Routine discharge process appropriate."
            )

        logger.info(f"Risk prediction: {risk_level} ({risk_score:.1%})")

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "interpretation": interpretation,
            "features_used": patient_profile,
        }


def extract_patient_profile_from_query(question: str) -> dict:
    """
    Simple keyword-based extractor to pull patient profile
    information from a natural language query.
    Returns a dict of features found in the question.
    """
    profile = {}
    question_lower = question.lower()

    # Age extraction
    import re
    age_match = re.search(r"(\d+)[- ]?year[s]?[- ]?old", question_lower)
    if age_match:
        profile["age"] = float(age_match.group(1))

    # Gender
    if any(w in question_lower for w in ["male", " man ", "mr.", "his "]):
        profile["gender_m"] = 1
    elif any(w in question_lower for w in ["female", "woman", "mrs.", "her "]):
        profile["gender_m"] = 0

    # Admission type
    if "emergency" in question_lower or "urgent" in question_lower:
        profile["admission_type_emergency"] = 1

    # Conditions
    condition_map = {
        "has_diabetes": ["diabetes", "diabetic", "hyperglycemia"],
        "has_heart_failure": ["heart failure", "chf", "cardiac failure"],
        "has_hypertension": ["hypertension", "high blood pressure"],
        "has_renal_disease": ["renal", "kidney disease", "ckd", "chronic kidney"],
        "has_pneumonia": ["pneumonia"],
        "has_sepsis": ["sepsis", "septic"],
        "has_copd": ["copd", "emphysema", "chronic obstructive"],
    }

    for feature, keywords in condition_map.items():
        if any(kw in question_lower for kw in keywords):
            profile[feature] = 1

    # Length of stay
    los_match = re.search(r"(\d+)[- ]?day[s]? (?:stay|admission|hospitali)", question_lower)
    if los_match:
        profile["los_days"] = float(los_match.group(1))

    return profile