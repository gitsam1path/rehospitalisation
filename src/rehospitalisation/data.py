"""
Séjours hospitaliers de patients diabétiques (130 hôpitaux américains, 1999-2008).

Cible : le patient est-il réhospitalisé dans les 30 jours suivant sa sortie ?
L'origine ethnique et le sexe ne sont PAS utilisés par le modèle : ils servent uniquement à l'audit d'équité.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "readmission.csv"

RENAME = {
    "readmitted": "rehospitalise", "race": "origine", "sex": "sexe", "age": "tranche_age",
    "admission_source": "mode_admission", "blood_glucose": "glycemie", "insurer": "assurance",
    "duration": "duree_sejour_jours", "n_previous_visits": "sejours_annee_precedente",
    "n_diagnoses": "nb_diagnostics", "n_procedures": "nb_actes", "n_medications": "nb_medicaments",
}
VALUES = {
    "origine": {"Caucasian": "Blanc", "African American": "Afro-américain", "Hispanic": "Hispanique",
                "Asian": "Asiatique", "Other": "Autre", "Unknown": "Inconnu"},
    "sexe": {"Female": "Femme", "Male": "Homme"},
    "mode_admission": {"Emergency": "Urgences", "Referral": "Adressé par un médecin", "Other": "Autre"},
    "glycemie": {"Very High": "Très élevée", "High": "Élevée", "Normal": "Normale"},
    "assurance": {"Medicare": "Medicare (public, +65 ans)", "Medicaid": "Medicaid (public, bas revenus)",
                  "Private": "Privée", "Self-Pay": "Sans assurance"},
}
SENSITIVE = ["origine", "sexe"]
NUMERIC = ["age", "duree_sejour_jours", "sejours_annee_precedente", "nb_diagnostics", "nb_actes", "nb_medicaments"]
CATEGORICAL = ["mode_admission", "glycemie", "assurance"]
TARGET = "rehospitalise"


def load(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=RENAME)
    for col, mapping in VALUES.items():
        df[col] = df[col].map(mapping)
    # valeurs manquantes informatives : la glycémie n'a pas été mesurée, l'assurance n'est pas renseignée
    df["glycemie"] = df["glycemie"].fillna("Non mesurée")
    df["assurance"] = df["assurance"].fillna("Non renseignée")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    # tranche d'âge « [70-80) » -> âge au milieu de la tranche (75)
    df["age"] = df["tranche_age"].str.extract(r"\[(\d+)-").astype(float)[0] + 5
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC + CATEGORICAL]
