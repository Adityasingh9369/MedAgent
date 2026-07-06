import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

MIMIC_DIR = Path("data/mimic")


def load_admissions() -> pd.DataFrame:
    df = pd.read_csv(MIMIC_DIR / "ADMISSIONS.csv")
    df.columns = df.columns.str.lower()
    for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_patients() -> pd.DataFrame:
    df = pd.read_csv(MIMIC_DIR / "PATIENTS.csv")
    df.columns = df.columns.str.lower()
    for col in ["dob", "dod"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_diagnoses() -> pd.DataFrame:
    df = pd.read_csv(MIMIC_DIR / "DIAGNOSES_ICD.csv")
    df.columns = df.columns.str.lower()
    return df


def load_icustays() -> pd.DataFrame:
    df = pd.read_csv(MIMIC_DIR / "ICUSTAYS.csv")
    df.columns = df.columns.str.lower()
    for col in ["intime", "outtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# def compute_age(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
#     merged = admissions.merge(patients[["subject_id", "dob", "gender"]], on="subject_id", how="left")
#     merged["age"] = (merged["admittime"] - merged["dob"]).dt.days / 365.25
#     # MIMIC masks ages > 89 as ~300 — clip to 90
#     merged["age"] = merged["age"].clip(upper=90)
#     return merged

def compute_age(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    merged = admissions.merge(patients[["subject_id", "dob", "gender"]], on="subject_id", how="left")
    
    # Calculate age using year difference to avoid pandas datetime64 nanosecond overflow
    merged["age"] = merged["admittime"].dt.year - merged["dob"].dt.year
    
    # MIMIC masks ages > 89 as ~300 — clip to 90
    merged["age"] = merged["age"].clip(upper=90)
    
    return merged


def build_readmission_label(admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Label = 1 if patient was readmitted within 30 days of discharge.
    For each admission, we look at whether the same patient had
    another admission within 30 days after discharge.
    """
    adm = admissions.sort_values(["subject_id", "admittime"]).copy()
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)
    adm["days_to_readmission"] = (
        adm["next_admittime"] - adm["dischtime"]
    ).dt.days
    adm["readmitted_30d"] = (
        (adm["days_to_readmission"] >= 0) &
        (adm["days_to_readmission"] <= 30)
    ).astype(int)
    return adm


def build_diagnosis_features(diagnoses: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Build binary flags for common condition groups using ICD-9 code prefixes.
    """
    # Count diagnoses per admission
    diag_count = diagnoses.groupby("hadm_id").size().reset_index(name="diagnosis_count")

    # Common ICD-9 prefixes
    conditions = {
        "has_diabetes": ["250"],
        "has_heart_failure": ["428"],
        "has_hypertension": ["401", "402", "403", "404", "405"],
        "has_renal_disease": ["585", "586", "403"],
        "has_pneumonia": ["486", "485", "481", "482", "483", "484"],
        "has_sepsis": ["038", "995"],
        "has_copd": ["491", "492", "496"],
    }

    hadm_ids = admissions["hadm_id"].unique()
    flags = pd.DataFrame({"hadm_id": hadm_ids})

    for flag_name, prefixes in conditions.items():
        matching = diagnoses[
            diagnoses["icd9_code"].astype(str).str.startswith(tuple(prefixes))
        ]["hadm_id"].unique()
        flags[flag_name] = flags["hadm_id"].isin(matching).astype(int)

    flags = flags.merge(diag_count, on="hadm_id", how="left")
    flags["diagnosis_count"] = flags["diagnosis_count"].fillna(0)
    return flags


def build_icu_features(icustays: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ICU stay features per admission.
    """
    icu = icustays.copy()
    icu["icu_los"] = icu["los"]  # already in days

    agg = icu.groupby("hadm_id").agg(
        num_icu_stays=("icustay_id", "count"),
        total_icu_los=("icu_los", "sum"),
        max_icu_los=("icu_los", "max"),
    ).reset_index()

    return agg


def build_feature_matrix() -> pd.DataFrame:
    """
    Main function — joins all tables and returns the final
    feature matrix with target label ready for model training.
    """
    logger.info("Loading MIMIC tables...")
    admissions = load_admissions()
    patients = load_patients()
    diagnoses = load_diagnoses()
    icustays = load_icustays()

    logger.info("Computing readmission labels...")
    admissions = build_readmission_label(admissions)

    logger.info("Computing patient age...")
    admissions = compute_age(patients, admissions)

    logger.info("Building diagnosis features...")
    diag_features = build_diagnosis_features(diagnoses, admissions)

    logger.info("Building ICU features...")
    icu_features = build_icu_features(icustays)

    logger.info("Joining all features...")
    df = admissions.merge(diag_features, on="hadm_id", how="left")
    df = df.merge(icu_features, on="hadm_id", how="left")

    # Admission type and insurance as categorical
    df["admission_type_emergency"] = (df["admission_type"] == "EMERGENCY").astype(int)
    df["gender_m"] = (df["gender"] == "M").astype(int)

    # Length of stay in days
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.days

    # Fill nulls
    df["num_icu_stays"] = df["num_icu_stays"].fillna(0)
    df["total_icu_los"] = df["total_icu_los"].fillna(0)
    df["max_icu_los"] = df["max_icu_los"].fillna(0)

    # In-hospital deaths — exclude from readmission prediction
    df = df[df["hospital_expire_flag"] == 0].copy()

    feature_cols = [
        "age", "gender_m", "admission_type_emergency",
        "los_days", "diagnosis_count",
        "has_diabetes", "has_heart_failure", "has_hypertension",
        "has_renal_disease", "has_pneumonia", "has_sepsis", "has_copd",
        "num_icu_stays", "total_icu_los", "max_icu_los",
    ]

    target_col = "readmitted_30d"

    final = df[feature_cols + [target_col, "hadm_id", "subject_id"]].copy()
    final = final.dropna(subset=feature_cols)

    logger.info(f"Feature matrix built: {len(final)} admissions, {final[target_col].sum()} readmissions")
    logger.info(f"Readmission rate: {final[target_col].mean():.2%}")

    return final