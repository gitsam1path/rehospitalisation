"""
Prédire le risque de réhospitalisation à 30 jours.

1. 5 modèles comparés par validation croisée (ROC AUC, PR AUC, score de Brier) ;
2. calibration : une probabilité de 20 % doit correspondre à 20 % de réhospitalisations réelles ;
3. décision : l'équipe de suivi ne peut appeler qu'une partie des patients -> qui choisir ?
4. audit d'équité selon l'origine et le sexe (variables NON utilisées par le modèle).

Lancer depuis src/ :  python -m rehospitalisation.train
"""

from __future__ import annotations

import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL, NUMERIC, ROOT, SENSITIVE, TARGET, features, load

warnings.filterwarnings("ignore")
REPORTS, FIG, MODELS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"
SEED = 42
CAPACITES = [0.05, 0.10, 0.20, 0.30]  # part des patients que l'équipe peut suivre
CAPACITE_REFERENCE = 0.10

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1a73e8", "#e8710a", "#188038", "#d93025", "#9aa0a6", "#8e6bbf"
COLORS = {"Référence (taux moyen)": GREY, "Régression logistique": BLUE, "Arbre de décision": PURPLE,
          "Forêt aléatoire": ORANGE, "LightGBM": GREEN}


def models() -> dict:
    onehot = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    ordinal = ColumnTransformer([("num", "passthrough", NUMERIC),
                                 ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)])
    return {
        "Référence (taux moyen)": DummyClassifier(strategy="prior"),
        "Régression logistique": make_pipeline(onehot, LogisticRegression(C=1.0, max_iter=2000)),
        "Arbre de décision": make_pipeline(onehot, DecisionTreeClassifier(max_depth=5, min_samples_leaf=200, random_state=SEED)),
        "Forêt aléatoire": make_pipeline(ordinal, RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5,
                                                                        n_jobs=-1, random_state=SEED)),
        "LightGBM": make_pipeline(ordinal, lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                                                              colsample_bytree=0.8, random_state=SEED, verbose=-1)),
    }


def at_capacity(y, score, capacity) -> dict:
    """Si l'on suit les `capacity` % de patients les plus à risque : combien de réhospitalisations cible-t-on ?"""
    n = int(round(len(score) * capacity))
    top = np.argsort(-score)[:n]
    caught = int(np.asarray(y)[top].sum())
    return {"patients_suivis": n, "rehospitalisations_ciblees": caught,
            "rappel_pct": caught / int(np.sum(y)) * 100, "precision_pct": caught / n * 100,
            "lift": (caught / n) / np.mean(y)}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    for d in (FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    df = load()
    train, test = train_test_split(df, test_size=0.25, stratify=df[TARGET], random_state=SEED)
    X, y, Xt, yt = features(train), train[TARGET], features(test), test[TARGET].to_numpy()
    print(f"{len(df)} séjours · taux de réhospitalisation {df[TARGET].mean():.1%} · "
          f"entraînement {len(train)}, test {len(test)}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, oof, test_scores, fitted = [], {}, {}, {}
    for name, m in models().items():
        p = cross_val_predict(clone(m), X, y, cv=cv, method="predict_proba")[:, 1]
        final = clone(m).fit(X, y)
        pt = final.predict_proba(Xt)[:, 1]
        oof[name], test_scores[name], fitted[name] = p, pt, final
        rows.append({"modele": name,
                     "cv_roc_auc": roc_auc_score(y, p), "cv_pr_auc": average_precision_score(y, p), "cv_brier": brier_score_loss(y, p),
                     "test_roc_auc": roc_auc_score(yt, pt), "test_pr_auc": average_precision_score(yt, pt),
                     "test_brier": brier_score_loss(yt, pt),
                     "test_rappel_top10_pct": at_capacity(yt, pt, CAPACITE_REFERENCE)["rappel_pct"]})
        print(f"  {name:24s} ROC AUC CV {rows[-1]['cv_roc_auc']:.3f} / test {rows[-1]['test_roc_auc']:.3f} · "
              f"PR AUC test {rows[-1]['test_pr_auc']:.3f} · rappel top 10 % {rows[-1]['test_rappel_top10_pct']:.1f} %")
    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best = table.iloc[0]["modele"]
    # Règle fixée à l'avance : si la régression logistique est à moins de 0,005 d'AUC du meilleur, on la préfère (plus simple, explicable)
    lr_auc = table.set_index("modele").loc["Régression logistique", "cv_roc_auc"]
    if best != "Régression logistique" and table.iloc[0]["cv_roc_auc"] - lr_auc < 0.005:
        best = "Régression logistique"
        reason = "à moins de 0,005 d'AUC du meilleur : on préfère le modèle le plus simple et explicable"
    else:
        reason = "meilleur ROC AUC en validation croisée"
    print(f"Modèle retenu : {best} ({reason})")

    # ─── Calibration ───
    calibrated = CalibratedClassifierCV(clone(models()[best]), method="isotonic", cv=5).fit(X, y)
    p_cal = calibrated.predict_proba(Xt)[:, 1]
    calib = {"brut": brier_score_loss(yt, test_scores[best]), "calibre": brier_score_loss(yt, p_cal)}

    # ─── Décision selon la capacité de suivi ───
    capacity = {f"{int(c * 100)} %": at_capacity(yt, p_cal, c) for c in CAPACITES}

    # ─── Audit d'équité (origine et sexe : jamais vus par le modèle) ───
    n_top = int(round(len(p_cal) * CAPACITE_REFERENCE))
    selected = np.zeros(len(p_cal), dtype=bool)
    selected[np.argsort(-p_cal)[:n_top]] = True
    audit = {}
    for attr in SENSITIVE:
        g = pd.DataFrame({"groupe": test[attr].to_numpy(), "y": yt, "sel": selected, "p": p_cal})
        res = g.groupby("groupe").apply(lambda d: pd.Series({
            "patients": len(d),
            "taux_reel_pct": d["y"].mean() * 100,
            "risque_moyen_predit_pct": d["p"].mean() * 100,
            "part_selectionnee_pct": d["sel"].mean() * 100,
            "rappel_pct": (d["sel"] & (d["y"] == 1)).sum() / max(1, (d["y"] == 1).sum()) * 100,
            "auc": roc_auc_score(d["y"], d["p"]) if d["y"].nunique() == 2 and len(d) > 200 else np.nan,
        }), include_groups=False)
        audit[attr] = res.round(2)
        print(f"\nAudit d'équité – {attr} :\n{res.round(1).to_string()}")

    # ─── Interprétation ───
    imp = permutation_importance(fitted[best], Xt, yt, scoring="roc_auc", n_repeats=8, random_state=SEED, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=Xt.columns).sort_values(ascending=False)
    # rapports de cotes de la régression logistique (toujours calculés : c'est le modèle le plus lisible)
    odds = {}
    if True:
        lr = fitted["Régression logistique"]
        coef = pd.Series(lr[-1].coef_[0], index=lr[0].get_feature_names_out())
        sd = X[NUMERIC].std()
        for f, label, delta in [("sejours_annee_precedente", "+1 séjour l'année précédente", 1),
                                ("nb_diagnostics", "+1 diagnostic", 1), ("duree_sejour_jours", "+1 jour d'hospitalisation", 1),
                                ("nb_medicaments", "+5 médicaments", 5), ("age", "+10 ans", 10)]:
            odds[label] = float(np.exp(coef[f"num__{f}"] / sd[f] * delta))
        # catégories : rapport de cotes par rapport à une catégorie de référence (sinon il ne se compare à rien)
        references = {"mode_admission": "Adressé par un médecin", "glycemie": "Non mesurée", "assurance": "Privée"}
        for var, ref in references.items():
            c_ref = coef[f"cat__{var}_{ref}"]
            for c in coef.index:
                prefix = f"cat__{var}_"
                if c.startswith(prefix) and c != f"cat__{var}_{ref}":
                    odds[f"{var} : {c[len(prefix):]} (vs {ref})"] = float(np.exp(coef[c] - c_ref))

    # ─── Figures ───
    fig_taux(df)
    fig_comparaison(table)
    fig_roc(yt, test_scores)
    fig_calibration(yt, test_scores[best], p_cal, best)
    fig_capacite(yt, p_cal)
    fig_importance(importance)
    fig_equite(audit)

    joblib.dump({"modele": calibrated, "variables": NUMERIC + CATEGORICAL, "nom": best}, MODELS / "model.joblib")
    report = {
        "donnees": {"sejours": len(df), "taux_rehospitalisation_pct": round(df[TARGET].mean() * 100, 2),
                    "entrainement": len(train), "test": len(test), "variables": NUMERIC + CATEGORICAL,
                    "variables_exclues_du_modele": SENSITIVE},
        "modele_retenu": best, "raison": reason,
        "resultats": json.loads(table.round(4).to_json(orient="records", force_ascii=False)),
        "calibration_brier": {k: round(v, 5) for k, v in calib.items()},
        "decision_selon_capacite": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in capacity.items()},
        "audit_equite": {k: json.loads(v.to_json(orient="index", force_ascii=False)) for k, v in audit.items()},
        "importance": {k: round(float(v), 4) for k, v in importance.items()},
        "rapports_de_cotes": {k: round(v, 3) for k, v in odds.items()},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nCapacité de suivi :", {k: (round(v['rappel_pct'], 1), round(v['lift'], 2)) for k, v in capacity.items()})
    print("Brier brut / calibré :", calib)
    print("Rapports de cotes :", {k: round(v, 2) for k, v in odds.items()})


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────
def fig_taux(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    base = df[TARGET].mean() * 100
    v = df.groupby(df["sejours_annee_precedente"].clip(upper=5))[TARGET].mean() * 100
    axes[0].bar([str(int(i)) if i < 5 else "5+" for i in v.index], v, color=BLUE)
    axes[0].set_title("Séjours l'année précédente")
    v = df.groupby("mode_admission")[TARGET].mean().sort_values() * 100
    axes[1].bar(v.index, v, color=ORANGE)
    axes[1].set_title("Mode d'admission")
    axes[1].tick_params(axis="x", labelsize=8)
    v = df.groupby(pd.cut(df["duree_sejour_jours"], [0, 2, 4, 7, 14], labels=["1-2 j", "3-4 j", "5-7 j", "8-14 j"]),
                   observed=True)[TARGET].mean() * 100
    axes[2].bar(v.index.astype(str), v, color=GREEN)
    axes[2].set_title("Durée du séjour")
    for ax in axes:
        ax.axhline(base, color="black", ls="--", lw=0.8)
    axes[0].set_ylabel("Réhospitalisés à 30 jours (%)")
    axes[0].text(0, base + 0.3, f"moyenne {base:.1f} %", fontsize=8)
    fig.suptitle("Le signal le plus fort : les hospitalisations passées", fontsize=11)
    _save(fig, "facteurs_de_risque")


def fig_comparaison(table):
    t = table.sort_values("test_roc_auc")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bars = ax.barh(t["modele"], t["test_roc_auc"], color=[COLORS[m] for m in t["modele"]])
    for b, (_, r) in zip(bars, t.iterrows()):
        ax.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2,
                f"ROC AUC {r['test_roc_auc']:.3f} · PR AUC {r['test_pr_auc']:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0.45, t["test_roc_auc"].max() + 0.12)
    ax.set_xlabel("ROC AUC sur le jeu de test (0,5 = hasard)")
    ax.set_title("Comparaison des modèles")
    ax.grid(axis="y", visible=False)
    _save(fig, "comparaison_modeles")


def fig_roc(yt, scores):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for name, s in scores.items():
        if name.startswith("Référence"):
            continue
        fpr, tpr, _ = roc_curve(yt, s)
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.6, label=f"{name} ({roc_auc_score(yt, s):.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Hasard (0,5)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs (rappel)")
    ax.set_title("Courbes ROC (jeu de test)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "courbes_roc")


def fig_calibration(yt, raw, cal, name):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for s, label, color in [(raw, f"{name} (brut)", GREY), (cal, f"{name} (calibré)", BLUE)]:
        frac, mean = calibration_curve(yt, s, n_bins=10, strategy="quantile")
        ax.plot(mean * 100, frac * 100, marker="o", color=color, label=label)
    lim = max(raw.max(), cal.max()) * 100 * 0.6
    ax.plot([0, lim], [0, lim], color="black", ls="--", lw=1, label="Calibration parfaite")
    ax.set_xlabel("Risque prédit (%)")
    ax.set_ylabel("Taux réel de réhospitalisation (%)")
    ax.set_title("Les probabilités sont-elles fiables ?")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "calibration")


def fig_capacite(yt, score):
    caps = np.linspace(0.01, 1, 100)
    recall = [at_capacity(yt, score, c)["rappel_pct"] for c in caps]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(caps * 100, recall, color=BLUE, lw=2.2, label="Avec le modèle")
    ax.plot(caps * 100, caps * 100, color=GREY, ls="--", label="Patients choisis au hasard")
    for c in (0.10, 0.20):
        r = at_capacity(yt, score, c)["rappel_pct"]
        ax.scatter([c * 100], [r], color=RED, zorder=3)
        ax.annotate(f"suivre {int(c * 100)} % des patients\n→ {r:.0f} % des réhospitalisations",
                    (c * 100, r), xytext=(c * 100 + 6, r - 12), fontsize=8.5, arrowprops=dict(arrowstyle="-", color=RED))
    ax.set_xlabel("Part des patients suivis après leur sortie (%)")
    ax.set_ylabel("Part des réhospitalisations ciblées (%)")
    ax.set_title("Qui appeler ? Le compromis selon la capacité de l'équipe")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _save(fig, "capacite_de_suivi")


def fig_importance(imp):
    labels = {"sejours_annee_precedente": "Séjours l'année précédente", "mode_admission": "Mode d'admission",
              "nb_diagnostics": "Nombre de diagnostics", "duree_sejour_jours": "Durée du séjour",
              "nb_medicaments": "Nombre de médicaments", "age": "Âge", "assurance": "Assurance",
              "nb_actes": "Nombre d'actes", "glycemie": "Glycémie"}
    s = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh([labels.get(i, i) for i in s.index], s.values, color=BLUE)
    ax.set_xlabel("Baisse du ROC AUC si l'on brouille la variable")
    ax.set_title("Les variables qui comptent")
    ax.grid(axis="y", visible=False)
    _save(fig, "importance_variables")


def fig_equite(audit):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1.2]})
    for ax, attr in zip(axes, ["origine", "sexe"]):
        a = audit[attr].sort_values("patients", ascending=False)
        x = np.arange(len(a))
        ax.bar(x - 0.2, a["taux_reel_pct"], width=0.4, color=GREY, label="Taux réel de réhospitalisation")
        ax.bar(x + 0.2, a["risque_moyen_predit_pct"], width=0.4, color=BLUE, label="Risque moyen prédit")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({int(n)})" for g, n in zip(a.index, a["patients"])], fontsize=8)
        ax.set_title(f"Selon {'l’origine' if attr == 'origine' else 'le sexe'}", fontsize=10)
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.7, -0.18))
    fig.suptitle("Audit d'équité : le risque prédit suit-il le risque réel dans chaque groupe ?", fontsize=11)
    _save(fig, "audit_equite")


if __name__ == "__main__":
    main()
