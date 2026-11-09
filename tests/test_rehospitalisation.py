import json
from pathlib import Path

import numpy as np
import pytest

from rehospitalisation.data import SENSITIVE, TARGET, features, load
from rehospitalisation.train import at_capacity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return load()


def test_donnees(df):
    assert len(df) == 71515
    assert 0.08 < df[TARGET].mean() < 0.10
    assert df[["glycemie", "assurance"]].isna().sum().sum() == 0   # « non mesurée » / « non renseignée »
    assert df["age"].between(5, 95).all()


def test_attributs_sensibles_exclus(df):
    """L'origine et le sexe ne doivent jamais entrer dans le modèle."""
    cols = set(features(df).columns)
    for attr in SENSITIVE:
        assert attr not in cols
    assert TARGET not in cols


def test_capacite_de_suivi():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    r = at_capacity(y, score, 0.3)           # on suit 3 patients sur 10
    assert r["patients_suivis"] == 3 and r["rehospitalisations_ciblees"] == 2
    assert r["rappel_pct"] == pytest.approx(200 / 3)
    assert r["lift"] == pytest.approx((2 / 3) / 0.3)


def test_rapport():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("lancer d'abord : cd src && python -m rehospitalisation.train")
    r = json.loads(path.read_text(encoding="utf-8"))
    res = {m["modele"]: m for m in r["resultats"]}
    assert res[r["modele_retenu"]]["test_roc_auc"] > 0.6
    assert res["Référence (taux moyen)"]["test_roc_auc"] == pytest.approx(0.5)
    assert r["calibration_brier"]["calibre"] <= r["calibration_brier"]["brut"] + 1e-3
    assert set(r["audit_equite"]) == set(SENSITIVE)
