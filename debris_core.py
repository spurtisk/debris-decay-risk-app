"""Pipeline and models for the space debris decay-risk project.

This is a Python (pandas + scikit-learn) mirror of the Phase 3 Spark notebooks
01 to 04. It applies the same cleaning rules, the same features, the same
80/20 stratified split idea, and the same three models. The Spark notebooks
remain the reference run, so numbers here can differ slightly from them
(see the notes in the app).
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS_DATE = pd.Timestamp("2026-09-26")
MU_EARTH = 398600.4418  # km^3 / s^2
EPS = 1e-4
SEED = 42
TRAIN_FRACTION = 0.80
HIGH_RISK_TOP_FRACTION = 0.10
PERIOD_TOLERANCE = 0.01  # 1 percent relative error allowed in the Kepler check

TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY"]
ORBITS = ["GEO", "HEO", "LEO", "MEO"]
COUNTRIES = ["CHINA", "EU", "INDIA", "JAPAN", "OTHER", "RUSSIA", "USA"]
CATS = {"type": TYPES, "orbit_class": ORBITS, "source_country": COUNTRIES}
FEATURE_NUMERIC = ["perigee_km", "eccentricity", "inclination_deg", "log_rcs", "object_age_years"]

RAW_REQUIRED = [
    "type", "object_name", "orbit_class", "apogee_km", "perigee_km", "inclination_deg",
    "source_country", "eccentricity", "semi_major_axis_km", "period_minutes", "mean_motion",
    "speed_km_s", "rcs_m2", "launch_date", "launch_year", "decay_risk", "object_id",
]
NUMERIC = [
    "apogee_km", "perigee_km", "inclination_deg", "eccentricity", "semi_major_axis_km",
    "period_minutes", "mean_motion", "speed_km_s", "rcs_m2", "launch_year", "decay_risk",
]
SILVER_COLUMNS = [
    "object_id", "type", "orbit_class", "source_country", "perigee_km", "apogee_km",
    "inclination_deg", "eccentricity", "semi_major_axis_km", "period_minutes", "rcs_m2",
    "log_rcs", "rcs_outlier", "launch_date", "object_age_years", "altitude_band",
    "decay_risk", "log_decay_risk", "high_decay_risk",
]


class DataError(ValueError):
    """Raised when the uploaded file cannot be used."""


# ---------------------------------------------------------------- bronze ----
def load_csv(data: bytes) -> pd.DataFrame:
    """Read the raw CSV as text, the way the Bronze notebook does."""
    try:
        raw = pd.read_csv(io.BytesIO(data), dtype=str, on_bad_lines="error")
    except pd.errors.ParserError as err:
        raise DataError(f"The file has a malformed line: {err}") from err
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in RAW_REQUIRED if c not in raw.columns]
    if missing:
        raise DataError("Missing columns: " + ", ".join(missing))
    return raw


# ---------------------------------------------------------------- silver ----
def clean_silver(raw: pd.DataFrame):
    """Apply the Phase 1 rules C0 to C14. Returns (silver, rejects, audit)."""
    df = raw.copy().reset_index(drop=True)
    audit: list[dict] = []
    rejects: list[pd.DataFrame] = []

    def log(step: str, rows_in: int, note: str) -> None:
        audit.append({"step": step, "rows_in": rows_in, "rows_out": len(df),
                      "rows_removed": rows_in - len(df), "note": note})

    def reject(mask: pd.Series, step: str, reason: str) -> None:
        nonlocal df
        if not mask.any():
            return
        bad = df[mask].copy()
        bad["reject_step"] = step
        bad["reject_reason"] = reason
        rejects.append(bad)
        df = df[~mask].reset_index(drop=True)

    n = len(df)
    log("C0 read CSV", n, "every line parsed with 17 fields")

    n = len(df)
    reject(df.duplicated(keep="first"), "C2", "exact_duplicate")
    log("C2 remove exact duplicate rows", n, "all columns equal")

    n = len(df)
    reject(df.duplicated("object_id", keep="first"), "C3", "duplicate_object_id")
    log("C3 unique object_id", n, "object_id is the catalog key")

    n = len(df)
    for col in ["type", "orbit_class", "source_country"]:
        df[col] = df[col].str.strip().str.upper()
    log("C4 clean category text", n, "trim and uppercase")

    n = len(df)
    df["type"] = df["type"].replace({"ROCKET_BODY": "ROCKET BODY", "RB": "ROCKET BODY",
                                     "R": "ROCKET BODY", "B": "ROCKET BODY"})
    shares = df["source_country"].value_counts(normalize=True)
    rare = shares[shares < 0.01].index
    df.loc[df["source_country"].isin(rare), "source_country"] = "OTHER"
    log("C5 merge synonyms and rare countries", n,
        "rocket body variants merged; countries under 1% become OTHER")

    n = len(df)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    log("C6 coerce numeric columns", n, "unparseable values become null")

    n = len(df)
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["launch_year"] = df["launch_date"].dt.year.fillna(df["launch_year"])
    log("C7 parse dates, reconcile year", n, "launch_year follows launch_date")

    n = len(df)
    reject(~df["orbit_class"].isin(ORBITS), "C8", "unknown_orbit")
    reject(~df["type"].isin(TYPES), "C8", "unknown_type")
    log("C8 orbit and type values present", n, "unknown orbit or type removed")

    n = len(df)
    imputed = int(df["rcs_m2"].isna().sum())
    df["rcs_m2"] = df["rcs_m2"].fillna(df.groupby("type")["rcs_m2"].transform("median"))
    reject(df["decay_risk"].isna(), "C9", "missing_decay_risk")
    log("C9 fill and reject missing values", n,
        f"{imputed} missing RCS values filled by type median")

    n = len(df)
    rules = [
        ("perigee_not_above_100km", df["perigee_km"] > 100),
        ("apogee_below_perigee", df["apogee_km"] >= df["perigee_km"]),
        ("eccentricity_out_of_range", df["eccentricity"].between(0, 0.9999)),
        ("inclination_out_of_range", df["inclination_deg"].between(0, 180)),
        ("decay_risk_out_of_range", df["decay_risk"].between(0, 1)),
        ("rcs_not_positive", df["rcs_m2"] > 0),
        ("launch_year_before_1957", df["launch_year"] >= 1957),
        ("launch_date_after_analysis_date", df["launch_date"] <= ANALYSIS_DATE),
    ]
    reason = pd.Series([None] * len(df), dtype=object)
    for name, ok in rules:
        fails = (~ok.fillna(False).astype(bool)) & reason.isna()
        reason[fails] = name
    reject_mask = reason.notna()
    if reject_mask.any():
        bad = df[reject_mask].copy()
        bad["reject_step"] = "C10"
        bad["reject_reason"] = reason[reject_mask].values
        rejects.append(bad)
        df = df[~reject_mask].reset_index(drop=True)
    log("C10 remove impossible values", n, "first failing rule recorded for each rejected row")

    n = len(df)
    calc_period = 2 * np.pi * np.sqrt(df["semi_major_axis_km"] ** 3 / MU_EARTH) / 60
    rel_err = (df["period_minutes"] - calc_period).abs() / calc_period
    worst = float(rel_err.max()) if len(df) else 0.0
    reject(~(rel_err <= PERIOD_TOLERANCE), "C11", "period_mismatch_kepler")
    log("C11 Kepler period check", n,
        f"largest relative period error {worst:.2e} (speed check is in the Spark notebook)")

    n = len(df)
    q1, q3 = df["rcs_m2"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["rcs_outlier"] = (df["rcs_m2"] < q1 - 1.5 * iqr) | (df["rcs_m2"] > q3 + 1.5 * iqr)
    df["log_rcs"] = np.log10(df["rcs_m2"])
    log("C12 outlier flag and log of RCS", n,
        f"{int(df['rcs_outlier'].sum())} IQR outliers kept and flagged")

    n = len(df)
    df["object_age_years"] = (ANALYSIS_DATE - df["launch_date"]).dt.days / 365.25
    base = (df["perigee_km"] // 100 * 100).astype(int)
    df["altitude_band"] = base.astype(str) + "-" + (base + 100).astype(str)
    df["log_decay_risk"] = np.log10(df["decay_risk"] + EPS)
    p90 = df["decay_risk"].quantile(1 - HIGH_RISK_TOP_FRACTION)
    df["high_decay_risk"] = (df["decay_risk"] >= p90).astype(int)
    log("C13 derived features", n,
        f"altitude band, log target, high-risk flag (top {int(HIGH_RISK_TOP_FRACTION*100)}%)")

    n = len(df)
    silver = df[SILVER_COLUMNS].reset_index(drop=True)
    log("C14 final Silver schema", n, "final columns")

    rejects_df = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(
        columns=["reject_step", "reject_reason"])
    return silver, rejects_df, pd.DataFrame(audit)


# ------------------------------------------------------------ features ----
def make_matrix(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """Numeric features plus one-hot columns. Linear models drop the first level."""
    cols = {c: df[c].astype(float).to_numpy() for c in FEATURE_NUMERIC}
    for col, cats in CATS.items():
        values = df[col].to_numpy()
        for cat in (cats[1:] if drop_first else cats):
            cols[f"{col}={cat}"] = (values == cat).astype(float)
    return pd.DataFrame(cols)


def stratified_split(silver: pd.DataFrame, train_fraction: float = TRAIN_FRACTION,
                     seed: int = SEED) -> np.ndarray:
    """Within each class, rank objects by a seeded hash of object_id; first share goes to train."""
    hashes = silver["object_id"].astype(str).map(
        lambda s: int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest()[:12], 16))
    is_train = np.zeros(len(silver), dtype=bool)
    for _, group in silver.assign(_h=hashes).groupby("high_decay_risk"):
        ordered = group.sort_values("_h")
        n_train = int(round(train_fraction * len(ordered)))
        is_train[silver.index.get_indexer(ordered.index[:n_train])] = True
    return is_train


# -------------------------------------------------------------- models ----
def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1e12, max_iter=5000))


def _forest(trees: int, depth: int, seed: int):
    return RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=3,
                                  class_weight="balanced", random_state=seed, n_jobs=-1)


def _classification(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "F1": f1_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    return metrics


@dataclass
class Fitted:
    lin: object
    lr: object
    rf: object
    lr_threshold: float
    rf_threshold: float
    test_frame: pd.DataFrame
    metrics: pd.DataFrame
    lr_coef: pd.DataFrame
    rf_importance: pd.DataFrame
    single_feature_auc: pd.Series
    n_train: int
    n_test: int


def fit_all(silver: pd.DataFrame, train_fraction: float = TRAIN_FRACTION, seed: int = SEED,
            rf_trees: int = 300, rf_depth: int = 10) -> Fitted:
    is_train = stratified_split(silver, train_fraction, seed)
    tr = silver[is_train].reset_index(drop=True)
    te = silver[~is_train].reset_index(drop=True)

    x_tr_lin, x_te_lin = make_matrix(tr, True).to_numpy(), make_matrix(te, True).to_numpy()
    x_tr_rf, x_te_rf = make_matrix(tr).to_numpy(), make_matrix(te).to_numpy()
    y_tr_log, y_te_log = tr["log_decay_risk"].to_numpy(), te["log_decay_risk"].to_numpy()
    y_tr_hi, y_te_hi = tr["high_decay_risk"].to_numpy(), te["high_decay_risk"].to_numpy()
    raw_te = te["decay_risk"].to_numpy()

    # Linear Regression on log10(decay_risk + EPS)
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(x_tr_lin, y_tr_log)
    pred_log = lin.predict(x_te_lin)
    pred_raw = np.clip(10 ** pred_log - EPS, 0, 1)
    lin_metrics = {
        "R2 log scale": r2_score(y_te_log, pred_log),
        "R2 original scale": r2_score(raw_te, pred_raw),
        "RMSE": float(np.sqrt(mean_squared_error(raw_te, pred_raw))),
        "MAE": mean_absolute_error(raw_te, pred_raw),
    }

    # Out-of-fold thresholds on training rows only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = {"lr": np.zeros(len(tr)), "rf": np.zeros(len(tr))}
    for a, b in skf.split(x_tr_lin, y_tr_hi):
        oof["lr"][b] = _logistic().fit(x_tr_lin[a], y_tr_hi[a]).predict_proba(x_tr_lin[b])[:, 1]
        oof["rf"][b] = _forest(rf_trees, rf_depth, seed).fit(x_tr_rf[a], y_tr_hi[a]).predict_proba(x_tr_rf[b])[:, 1]
    lr_thr = _best_f1_threshold(y_tr_hi, oof["lr"])
    rf_thr = _best_f1_threshold(y_tr_hi, oof["rf"])

    lr = _logistic().fit(x_tr_lin, y_tr_hi)
    rf = _forest(rf_trees, rf_depth, seed).fit(x_tr_rf, y_tr_hi)
    p_lr = lr.predict_proba(x_te_lin)[:, 1]
    p_rf = rf.predict_proba(x_te_rf)[:, 1]

    metrics = pd.DataFrame([
        {"model": "Linear Regression", "metric": k, "value": v} for k, v in lin_metrics.items()
    ] + [
        {"model": "Logistic Regression", "metric": k, "value": v}
        for k, v in _classification(y_te_hi, p_lr, lr_thr).items()
    ] + [
        {"model": "Random Forest", "metric": k, "value": v}
        for k, v in _classification(y_te_hi, p_rf, rf_thr).items()
    ])
    metrics.attrs["lr_threshold"] = lr_thr
    metrics.attrs["rf_threshold"] = rf_thr

    lr_coef = pd.DataFrame({
        "feature": make_matrix(tr, True).columns,
        "coefficient": lr[-1].coef_[0],
    }).sort_values("coefficient", key=np.abs, ascending=False)

    rf_imp = pd.DataFrame({
        "feature": make_matrix(tr).columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    single = pd.Series({c: roc_auc_score(y_te_hi, te[c].to_numpy()) for c in FEATURE_NUMERIC},
                       name="ROC-AUC alone")

    test_frame = te.assign(p_lr=p_lr, p_rf=p_rf, pred_log_risk=pred_log, pred_risk=pred_raw)
    return Fitted(lin=lin, lr=lr, rf=rf, lr_threshold=lr_thr, rf_threshold=rf_thr,
                  test_frame=test_frame, metrics=metrics, lr_coef=lr_coef,
                  rf_importance=rf_imp, single_feature_auc=single,
                  n_train=len(tr), n_test=len(te))


def predict_one(fitted: Fitted, inputs: dict) -> dict:
    """Score one object. inputs use the raw column names, with launch_date as a date."""
    rcs = float(inputs["rcs_m2"])
    launch = pd.Timestamp(inputs["launch_date"])
    row = pd.DataFrame([{
        "perigee_km": float(inputs["perigee_km"]),
        "eccentricity": float(inputs["eccentricity"]),
        "inclination_deg": float(inputs["inclination_deg"]),
        "log_rcs": float(np.log10(rcs)),
        "object_age_years": (ANALYSIS_DATE - launch).days / 365.25,
        "type": inputs["type"],
        "orbit_class": inputs["orbit_class"],
        "source_country": inputs["source_country"],
    }])
    x_lin = make_matrix(row, True).to_numpy()
    x_rf = make_matrix(row).to_numpy()
    risk = float(np.clip(10 ** fitted.lin.predict(x_lin)[0] - EPS, 0, 1))
    p_lr = float(fitted.lr.predict_proba(x_lin)[0, 1])
    p_rf = float(fitted.rf.predict_proba(x_rf)[0, 1])
    return {
        "predicted_decay_risk": risk,
        "p_high_logistic": p_lr,
        "p_high_forest": p_rf,
        "class_logistic": "high risk" if p_lr >= fitted.lr_threshold else "normal",
        "class_forest": "high risk" if p_rf >= fitted.rf_threshold else "normal",
    }


# --------------------------------------------------------------- gold ----
def gold_tables(silver: pd.DataFrame):
    summary = (silver.groupby(["altitude_band", "orbit_class", "type"])
               .agg(objects=("object_id", "count"),
                    median_decay_risk=("decay_risk", "median"),
                    high_risk_share=("high_decay_risk", "mean"),
                    mean_log_rcs=("log_rcs", "mean"))
               .reset_index())
    by_country = (silver.groupby("source_country")
                  .agg(objects=("object_id", "count"),
                       high_risk_objects=("high_decay_risk", "sum"),
                       median_decay_risk=("decay_risk", "median"))
                  .reset_index())
    return summary, by_country


# ------------------------------------------------------------ run all ----
def run_all(data: bytes, train_fraction: float = TRAIN_FRACTION, seed: int = SEED,
            rf_trees: int = 300, rf_depth: int = 10):
    raw = load_csv(data)
    silver, rejects, audit = clean_silver(raw)
    if len(silver) < 100:
        raise DataError("Too few rows remain after cleaning to train the models.")
    fitted = fit_all(silver, train_fraction, seed, rf_trees, rf_depth)
    summary, by_country = gold_tables(silver)
    return {
        "raw_rows": len(raw), "raw_columns": len(raw.columns),
        "silver": silver, "rejects": rejects, "audit": audit,
        "gold_summary": summary, "gold_country": by_country,
        "fitted": fitted,
    }
