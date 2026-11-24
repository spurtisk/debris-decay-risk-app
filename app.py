"""Space debris decay-risk: interactive walkthrough of the Phase 3 project.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy it on Streamlit Community Cloud.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import debris_core as dc

st.set_page_config(page_title="Space Debris Decay Risk", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    h1 { font-size: 2.1rem !important; }
    .muted { color: var(--text-color, #666); opacity: 0.75; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Reference numbers from the Spark notebooks (Phase 3 run), shown next to live results.
SPARK_REFERENCE = {
    "Linear Regression": {"R2 log scale": 0.9916, "R2 original scale": 0.9860,
                          "RMSE": 0.0076, "MAE": 0.0023},
    "Logistic Regression": {"ROC-AUC": 0.9994, "PR-AUC": 0.9960, "F1": 0.9843,
                            "Precision": 0.9895, "Recall": 0.9792,
                            "TP": 94, "FP": 1, "FN": 2, "TN": 865},
    "Random Forest": {"ROC-AUC": 0.9938, "PR-AUC": 0.9886, "F1": 0.9684,
                      "Precision": 0.9787, "Recall": 0.9583,
                      "TP": 92, "FP": 2, "FN": 4, "TN": 864},
}

DEFAULT_CSV = Path(__file__).parent / "data" / "space_debris_orbits_2026.csv"


# ----------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload the Kaggle CSV", type="csv",
                                help="space_debris_orbits_2026.csv from Kaggle, the same file the Spark notebooks used.")
    st.header("Training settings")
    train_fraction = st.slider("Training share", 0.60, 0.90, 0.80, 0.05,
                               help="Share of each class that goes to training. The Spark run used 0.80.")
    seed = st.number_input("Split seed", min_value=0, max_value=9999, value=42, step=1,
                           help="Changes which objects land in train and test.")
    rf_trees = st.select_slider("Random Forest trees", options=[50, 100, 200, 300], value=300)
    rf_depth = st.select_slider("Random Forest max depth", options=[5, 10, 15, 20], value=10)
    st.caption("Changing a setting reruns the pipeline and models. Results are cached per setting.")

if uploaded is not None:
    data_bytes = uploaded.getvalue()
    source_label = f"Uploaded file: {uploaded.name}"
elif DEFAULT_CSV.exists():
    data_bytes = DEFAULT_CSV.read_bytes()
    source_label = "Bundled file: data/space_debris_orbits_2026.csv"
else:
    st.title("Space Debris Decay Risk")
    st.info("Upload space_debris_orbits_2026.csv in the sidebar to run the project. "
            "Or place the file at data/space_debris_orbits_2026.csv in the repo and it loads automatically.")
    st.stop()


@st.cache_resource(show_spinner="Running the pipeline and training the three models...")
def run(data: bytes, frac: float, rng_seed: int, trees: int, depth: int):
    return dc.run_all(data, frac, rng_seed, trees, depth)


try:
    result = run(data_bytes, float(train_fraction), int(seed), int(rf_trees), int(rf_depth))
except dc.DataError as err:
    st.error(f"The file could not be used: {err}")
    st.stop()
except Exception as err:  # keep the app alive and show the reason
    st.error(f"Unexpected error while running the pipeline: {err}")
    st.stop()

silver = result["silver"]
rejects = result["rejects"]
audit = result["audit"]
fitted = result["fitted"]
high_total = int(silver["high_decay_risk"].sum())

# ----------------------------------------------------------- header ----
st.title("Space Debris Decay Risk")
st.caption(f"{source_label}. Cleaning, features, split and three models run live from the file you give it.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw records", f"{result['raw_rows']:,}", f"{result['raw_columns']} columns")
c2.metric("After cleaning", f"{len(silver):,}", f"-{len(rejects):,} rejected", delta_color="inverse")
c3.metric("High-risk objects", f"{high_total:,}", f"top {int(dc.HIGH_RISK_TOP_FRACTION*100)}% of decay risk")
logistic_auc = float(fitted.metrics.loc[(fitted.metrics["model"] == "Logistic Regression")
                                        & (fitted.metrics["metric"] == "ROC-AUC"), "value"].iat[0])
c4.metric("Logistic test ROC-AUC", f"{logistic_auc:.4f}")

tab_pipe, tab_models, tab_predict, tab_explore, tab_notes = st.tabs(
    ["Pipeline", "Models", "Predict", "Explore", "Notes"])

# --------------------------------------------------------- pipeline ----
with tab_pipe:
    st.subheader("From raw CSV to model-ready features")
    st.markdown(
        "The pipeline has four stages. Each stage produces a table, and the next stage reads it. "
        "The Spark notebooks write these as Delta tables. This app keeps them as in-memory tables with the same rules."
    )
    stage = pd.DataFrame([
        {"stage": "Bronze", "what": "Raw copy, every column kept as text", "rows": result["raw_rows"]},
        {"stage": "Silver", "what": "Cleaning rules C0 to C14 applied", "rows": len(silver)},
        {"stage": "Silver rejects", "what": "Rows removed, with the rule that removed them", "rows": len(rejects)},
        {"stage": "Gold features", "what": "Model inputs, labels and the train/test split", "rows": len(silver)},
    ])
    st.dataframe(stage, hide_index=True, width="stretch")

    st.subheader("Cleaning audit")
    st.caption("Each row is one step: the rows it received, the rows it kept, and what it did.")
    st.dataframe(audit, hide_index=True, width="stretch")

    if len(rejects):
        st.subheader("Why rows were rejected")
        reasons = (rejects.groupby(["reject_step", "reject_reason"]).size()
                   .rename("rows").reset_index())
        st.dataframe(reasons, hide_index=True, width="stretch")

    st.subheader("Split")
    test_high = int(fitted.test_frame["high_decay_risk"].sum())
    split_df = pd.DataFrame({
        "split": ["train", "test"],
        "objects": [fitted.n_train, fitted.n_test],
        "high-risk objects": [high_total - test_high, test_high],
    })
    st.dataframe(split_df, hide_index=True, width="stretch")
    st.caption("Stratified on the high-risk label. Each class is ranked by a seeded hash of object_id, "
               "and the first share goes to training.")

    st.subheader("Gold summaries")
    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("**By altitude band, orbit and type**")
        st.dataframe(result["gold_summary"], hide_index=True, width="stretch")
    with g2:
        st.markdown("**By source country**")
        st.dataframe(result["gold_country"], hide_index=True, width="stretch")

# ----------------------------------------------------------- models ----
with tab_models:
    st.subheader("Test-set results")
    st.caption("Live results on this run's test split, next to the Spark notebook values. "
               "Differences come from the split and settings, not from a bug. Changing the sidebar settings moves the live column.")

    rows = []
    for model, ref in SPARK_REFERENCE.items():
        live = fitted.metrics[fitted.metrics["model"] == model].set_index("metric")["value"]
        for metric, ref_value in ref.items():
            live_value = float(live.get(metric, np.nan))
            rows.append({"model": model, "metric": metric, "this run": live_value,
                         "Spark reference": ref_value,
                         "difference": live_value - ref_value})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={
                     "this run": st.column_config.NumberColumn(format="%.4f"),
                     "Spark reference": st.column_config.NumberColumn(format="%.4f"),
                     "difference": st.column_config.NumberColumn(format="%+.4f"),
                 })

    st.markdown(f"Classification threshold chosen from out-of-fold training predictions: "
                f"logistic **{fitted.lr_threshold:.3f}**, random forest **{fitted.rf_threshold:.3f}**. "
                f"The Spark run used 0.935 and 0.660.")

    cm1, cm2 = st.columns(2)
    for col, model in [(cm1, "Logistic Regression"), (cm2, "Random Forest")]:
        m = fitted.metrics[fitted.metrics["model"] == model].set_index("metric")["value"]
        with col:
            st.markdown(f"**{model}: confusion matrix**")
            cm = pd.DataFrame(
                [[int(m["TN"]), int(m["FP"])], [int(m["FN"]), int(m["TP"])]],
                index=["actual normal", "actual high risk"],
                columns=["predicted normal", "predicted high risk"],
            )
            st.dataframe(cm, width="stretch")

    st.subheader("What drives the predictions")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**Random Forest importance**")
        imp = fitted.rf_importance.head(10)
        st.bar_chart(imp, x="feature", y="importance", horizontal=True, height=340)
    with f2:
        st.markdown("**Each feature on its own (ROC-AUC on test)**")
        single = fitted.single_feature_auc.sort_values(ascending=False).rename_axis("feature").reset_index()
        st.bar_chart(single, x="feature", y="ROC-AUC alone", horizontal=True, height=340)
    st.caption("A value near 0.5 means the feature alone carries no signal. Log RCS alone almost separates the classes, "
               "which is why the models score so high.")

    st.markdown("**Logistic Regression coefficients (standardized inputs)**")
    st.dataframe(fitted.lr_coef.head(15), hide_index=True, width="stretch",
                 column_config={"coefficient": st.column_config.NumberColumn(format="%.3f")})

# ---------------------------------------------------------- predict ----
with tab_predict:
    st.subheader("Score one object")
    st.caption("Change any input and the three models update. The reference date for object age is the project's analysis date.")

    p1, p2, p3 = st.columns(3)
    with p1:
        obj_type = st.selectbox("Type", dc.TYPES, index=1)
        orbit = st.selectbox("Orbit class", dc.ORBITS, index=2)
        country = st.selectbox("Source country", dc.COUNTRIES, index=0)
    with p2:
        perigee = st.number_input("Perigee (km)", min_value=101.0, max_value=60000.0, value=550.0, step=10.0)
        ecc = st.number_input("Eccentricity", min_value=0.0, max_value=0.9999, value=0.0005, step=0.0001, format="%.4f")
        incl = st.number_input("Inclination (deg)", min_value=0.0, max_value=180.0, value=53.0, step=0.5)
    with p3:
        rcs_default = float(silver["rcs_m2"].median())
        rcs = st.number_input("Radar cross-section (m²)", min_value=0.0001, max_value=100.0,
                              value=round(rcs_default, 3), step=0.05, format="%.4f")
        launch = st.date_input("Launch date", value=date(2020, 1, 1),
                               min_value=date(1957, 10, 4), max_value=dc.ANALYSIS_DATE.date())

    if perigee <= 100:
        st.error("Perigee must be above 100 km.")
    else:
        out = dc.predict_one(fitted, {
            "type": obj_type, "orbit_class": orbit, "source_country": country,
            "perigee_km": perigee, "eccentricity": ecc, "inclination_deg": incl,
            "rcs_m2": rcs, "launch_date": launch,
        })
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted decay risk (Linear Regression)", f"{out['predicted_decay_risk']:.4f}",
                  help="Back-transformed from the log scale, clipped to 0 to 1.")
        r2.metric("Logistic: probability of high risk", f"{out['p_high_logistic']:.3f}",
                  out["class_logistic"], delta_color="off")
        r3.metric("Random Forest: probability of high risk", f"{out['p_high_forest']:.3f}",
                  out["class_forest"], delta_color="off")

        st.markdown("**How the risk changes with radar cross-section**")
        st.caption("Other inputs held as above. This shows how much the model depends on RCS.")
        grid = np.logspace(np.log10(silver["rcs_m2"].min()), np.log10(silver["rcs_m2"].max()), 60)
        rows = []
        for g in grid:
            o = dc.predict_one(fitted, {
                "type": obj_type, "orbit_class": orbit, "source_country": country,
                "perigee_km": perigee, "eccentricity": ecc, "inclination_deg": incl,
                "rcs_m2": g, "launch_date": launch,
            })
            rows.append({"rcs_m2": round(float(g), 4), "logistic": o["p_high_logistic"],
                         "random forest": o["p_high_forest"]})
        sens = pd.DataFrame(rows).set_index("rcs_m2")
        st.line_chart(sens, height=320)

    st.info("This is a model trained on one dataset, which Phase 1 describes as synthetic. "
            "Do not use it for real collision or re-entry decisions.")

# ---------------------------------------------------------- explore ----
with tab_explore:
    st.subheader("Explore the cleaned catalog")
    f_type, f_orbit, f_high = st.columns([2, 2, 1])
    sel_type = f_type.multiselect("Type", dc.TYPES, default=dc.TYPES)
    sel_orbit = f_orbit.multiselect("Orbit class", dc.ORBITS, default=dc.ORBITS)
    only_high = f_high.checkbox("High risk only", value=False)

    view = silver[silver["type"].isin(sel_type) & silver["orbit_class"].isin(sel_orbit)]
    if only_high:
        view = view[view["high_decay_risk"] == 1]
    st.write(f"{len(view):,} objects match.")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Decay risk distribution**")
        counts, edges = np.histogram(view["decay_risk"], bins=30, range=(0, 1))
        hist = pd.DataFrame({"decay risk": [round(float(e), 2) for e in edges[:-1]], "objects": counts})
        st.bar_chart(hist, x="decay risk", y="objects", height=300)
    with e2:
        st.markdown("**Perigee against decay risk**")
        sample = view.sample(min(len(view), 1500), random_state=0) if len(view) else view
        st.scatter_chart(sample, x="perigee_km", y="decay_risk", color="type", height=300)

    st.markdown("**Rows (first 500)**")
    st.dataframe(view.head(500), hide_index=True, width="stretch")
    st.download_button("Download the cleaned Silver table (CSV)",
                       data=silver.to_csv(index=False).encode("utf-8"),
                       file_name="silver_cleaned.csv", mime="text/csv")

# ------------------------------------------------------------ notes ----
with tab_notes:
    st.subheader("What this app is and is not")
    st.markdown(
        """
- **It is the project, re-run in Python.** The cleaning rules, features, split idea and three models follow the Phase 3 notebooks. The Spark notebooks remain the reference run.
- **Numbers can differ from the Spark run.** The split uses a hash the notebook does not use, and the Random Forest is scikit-learn's, not Spark's. Compare the live and reference columns on the Models tab.
- **Checks.** The Kepler period check runs here. The vis-viva speed check runs only in the Spark notebook.
- **The data is probably synthetic.** Phase 1 describes the file that way, so very high scores may not carry over to real catalogs.
- **Not for operational use.** Do not use the predictions for real collision or re-entry decisions.
        """
    )
    st.subheader("Run order in the Spark project")
    st.code("01_bronze_layer.ipynb  ->  02_silver_layer.ipynb  ->  03_gold_layer.ipynb  ->  04_mllib_models.ipynb",
            language="text")
