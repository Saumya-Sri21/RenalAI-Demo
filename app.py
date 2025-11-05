"""
Streamlit app: CKD classification + staging (single-file prototype)
Features:
- Upload your CSV or use demo dataset
- Preprocessing (imputation, encoding, scaling)
- Train model (XGBoost if available, else RandomForest)
- Show probability of CKD and map to CKD stage (using eGFR if present or probability thresholds)
- Simple explainability: SHAP (if installed) else feature importances
- Download trained model (pickle)
"""

import streamlit as st
st.set_page_config(page_title="CKD Classifier (Demo)", layout="wide")

import pandas as pd, numpy as np, io, pickle, base64
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

# optional libs
try:
    import xgboost as xgb
    XGBOOST_INSTALLED = True
except Exception:
    XGBOOST_INSTALLED = False

try:
    import shap
    SHAP_INSTALLED = True
except Exception:
    SHAP_INSTALLED = False

# -------------------------
# Helpers
# -------------------------
def compute_egfr_mdrd(creatinine_mg_dl, age, female: bool, black: bool=False):
    # MDRD simplified: eGFR = 175 * Scr^-1.154 * Age^-0.203 * 0.742 (if female) * 1.212 (if black)
    # Works for adults; returns NaN if input invalid
    try:
        scr = float(creatinine_mg_dl)
        ag = float(age)
        val = 175 * (scr ** -1.154) * (ag ** -0.203)
        if female: val *= 0.742
        if black: val *= 1.212
        return round(val, 1)
    except Exception:
        return np.nan

def stage_from_egfr(egfr):
    # CKD stages based on eGFR (ml/min/1.73m2)
    if np.isnan(egfr): return "Unknown"
    egfr = float(egfr)
    if egfr >= 90: return "Stage 1 (Normal or high)"
    if 60 <= egfr < 90: return "Stage 2 (Mild)"
    if 30 <= egfr < 60: return "Stage 3 (Moderate)"
    if 15 <= egfr < 30: return "Stage 4 (Severe)"
    if egfr < 15: return "Stage 5 (Kidney failure)"
    return "Unknown"

def stage_from_probability(p):
    # fallback mapping: adjust if you have clinical guidance
    if p < 0.2: return "No CKD (low risk)"
    if 0.2 <= p < 0.45: return "Possible CKD (monitor)"
    if 0.45 <= p < 0.7: return "Likely CKD (consult doctor)"
    return "Very likely CKD (urgent evaluation)"

def get_download_link_bytes(obj_bytes, filename, text):
    b64 = base64.b64encode(obj_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href

# -------------------------
# Sidebar: controls
# -------------------------
st.sidebar.title("Settings / Data")
mode = st.sidebar.radio("Mode", ["Use demo dataset", "Upload CSV"])
show_shap = st.sidebar.checkbox("Show SHAP (if available)", value=False)
oversample = st.sidebar.checkbox("Use SMOTE for class imbalance", value=True)
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.4, 0.2)
random_state = 42

# -------------------------
# Demo dataset (small sample)
# -------------------------
def make_demo_df(n=500, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.normal(55, 12, n).clip(18, 90).astype(int)
    female = rng.binomial(1, 0.45, n)
    sbp = rng.normal(135, 18, n).clip(90, 220)
    dbp = rng.normal(82, 12, n).clip(50, 140)
    creat = rng.normal(1.3, 0.7, n).clip(0.4, 6.0)
    proteinuria = rng.choice([0, 1, 2], size=n, p=[0.7,0.2,0.1])
    diabetes = rng.binomial(1, 0.25, n)
    hgb = rng.normal(13.5, 1.8, n).clip(7,18)
    # Build an underlying CKD risk score
    risk_score = (0.04*(age-50) + 0.8*(creat-1) + 0.6*proteinuria + 0.9*diabetes - 0.03*(hgb-13))
    prob_ckd = 1/(1+np.exp(-risk_score))
    y = (prob_ckd > 0.5).astype(int)
    df = pd.DataFrame({
        "age": age,
        "female": female,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "serum_creatinine": np.round(creat,3),
        "proteinuria": proteinuria,
        "diabetes": diabetes,
        "hemoglobin": np.round(hgb,2),
        "label_ckd": y
    })
    # Add eGFR estimated for some rows
    df["eGFR_mdrd"] = df.apply(lambda r: compute_egfr_mdrd(r["serum_creatinine"], r["age"], bool(r["female"])), axis=1)
    return df

# -------------------------
# Load dataset
# -------------------------
if mode == "Use demo dataset":
    df = make_demo_df(600)
    st.sidebar.success("Loaded demo dataset (synthetic).")
else:
    st.sidebar.markdown("Upload a CSV with at least these columns (or similar):")
    st.sidebar.markdown("`age, female(0/1), serum_creatinine, systolic_bp, diastolic_bp, proteinuria, diabetes(0/1), hemoglobin, label_ckd`")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {uploaded.name} — shape {df.shape}")
    else:
        st.info("Upload a CSV or switch to demo dataset to try the app.")
        st.stop()

st.title("CKD Classifier — Streamlit single-file prototype")
st.write("Upload data or use demo. Train model and get probability + staging. This is an academic prototype — not a medical device.")

# Show data preview
with st.expander("Dataset preview"):
    st.dataframe(df.head(10))

# -------------------------
# Column selection & target
# -------------------------
st.subheader("Data & target")
all_cols = list(df.columns)
target_default = "label_ckd" if "label_ckd" in df.columns else all_cols[-1]
target = st.selectbox("Target column (label)", all_cols, index=all_cols.index(target_default))
st.write("Selected target:", target)

# Feature suggestion: auto-detect numeric and categorical
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != target]
cat_cols = [c for c in df.columns if c not in numeric_cols + [target]]
st.write("Numeric features detected:", numeric_cols)
st.write("Categorical features detected:", cat_cols)

# allow user to select features
features = st.multiselect("Select feature columns to use for training", numeric_cols + cat_cols, default=numeric_cols + cat_cols)
if len(features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

# -------------------------
# Preprocessing + pipeline
# -------------------------
st.subheader("Preprocessing & Model")
num_feats = [c for c in features if c in numeric_cols]
cat_feats = [c for c in features if c in cat_cols]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_feats),
    ("cat", cat_pipeline, cat_feats)
])

# Choose model
model_choice = st.radio("Model", ["Auto (XGBoost if installed)", "RandomForest"])
if model_choice.startswith("Auto") and XGBOOST_INSTALLED:
    st.write("XGBoost detected — will use XGBoost classifier.")
    def make_model():
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, verbosity=0)
else:
    st.write("Using RandomForestClassifier.")
    def make_model():
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight="balanced")

# Option: basic hyperparam tuning (simple)
do_tune = st.checkbox("Do simple randomized hyperparameter tuning (can take longer)", value=False)

# -------------------------
# Train / Evaluate
# -------------------------
if st.button("Train model"):
    X = df[features].copy()
    y = df[target].copy()
    # drop rows with missing target
    mask = y.notna()
    X = X[mask]; y = y[mask]
    # map y to 0/1 if not numeric
    if y.dtype != int and y.dtype != float:
        y = pd.factorize(y)[0]
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    # oversample if requested
    if oversample:
        try:
            sm = SMOTE(random_state=random_state)
            X_res, y_res = sm.fit_resample(X_train.fillna(0), y_train)
            X_train = pd.DataFrame(X_res, columns=X_train.columns)
            y_train = y_res
            st.write("Applied SMOTE to training set — new class counts:", np.bincount(y_train))
        except Exception as e:
            st.warning("SMOTE failed — continuing without oversampling. " + str(e))

    # build pipeline
    model = make_model()
    full_pipeline = Pipeline([("preproc", preprocessor), ("clf", model)])

    # (optional) simple tuning: grid over a very small set
    if do_tune:
        from sklearn.model_selection import RandomizedSearchCV
        st.write("Starting randomized search (quick)...")
        if isinstance(model, RandomForestClassifier) or not XGBOOST_INSTALLED:
            param_dist = {"clf__n_estimators": [100,200,400], "clf__max_depth":[None,6,12], "clf__min_samples_split":[2,5]}
        else:
            param_dist = {"clf__n_estimators":[100,200], "clf__max_depth":[3,6,9], "clf__learning_rate":[0.01,0.1]}
        rs = RandomizedSearchCV(full_pipeline, param_dist, n_iter=6, cv=3, scoring="roc_auc", n_jobs=-1, random_state=random_state)
        rs.fit(X_train, y_train)
        st.write("Best params:", rs.best_params_)
        best = rs.best_estimator_
        full_pipeline = best
    else:
        full_pipeline.fit(X_train, y_train)

    # predictions
    y_proba = full_pipeline.predict_proba(X_test)[:,1]
    y_pred = full_pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    st.metric("ROC AUC (test)", f"{auc:.3f}")

    st.subheader("Classification report (test set)")
    st.text(classification_report(y_test, y_pred, digits=3))
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # feature importance / explainability
    st.subheader("Explainability")
    # try shap if requested and available
    if show_shap and SHAP_INSTALLED:
        st.write("Computing SHAP values (may take time)...")
        try:
            # Need to transform training data to model input
            X_train_trans = full_pipeline.named_steps["preproc"].transform(X_train)
            explainer = shap.Explainer(full_pipeline.named_steps["clf"], X_train_trans)
            X_test_trans = full_pipeline.named_steps["preproc"].transform(X_test)
            shap_values = explainer(X_test_trans)
            st.pyplot(shap.plots.bar(shap_values, max_display=10, show=False))
        except Exception as e:
            st.warning("SHAP failed: " + str(e))
    else:
        # fallback: permutation importance on pipeline
        st.write("Showing permutation importances (approx).")
        try:
            r = permutation_importance(full_pipeline, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
            imp_idx = np.argsort(r.importances_mean)[::-1][:10]
            imp_df = pd.DataFrame({
                "feature": np.array(num_feats + list(full_pipeline.named_steps["preproc"].transformers_[1][1].named_steps["onehot"].get_feature_names_out(cat_feats) if len(cat_feats)>0 else []))[imp_idx],
                "importance": r.importances_mean[imp_idx]
            })
            st.table(imp_df)
        except Exception as e:
            st.warning("Permutation importance failed: " + str(e))

    # Save pipeline to session state for predictions
    st.session_state["model_pipeline"] = full_pipeline
    st.session_state["features"] = features
    st.success("Model trained and saved in session. Use the 'Predict' section below.")

    # allow download
    buf = io.BytesIO()
    pickle.dump(full_pipeline, buf)
    buf.seek(0)
    b = buf.read()
    st.markdown(get_download_link_bytes(b, "ckd_model.pkl", "Download trained model (.pkl)"), unsafe_allow_html=True)

# -------------------------
# Prediction UI
# -------------------------
st.subheader("Predict single patient")
if "model_pipeline" not in st.session_state:
    st.info("Train a model first (or reload app with a pre-trained model).")
else:
    model_pipe = st.session_state["model_pipeline"]
    feat_names = st.session_state["features"]

    # Create input widgets dynamically for features
    input_data = {}
    cols = st.columns(3)
    for i, f in enumerate(feat_names):
        col = cols[i % 3]
        sample_val = df[f].dropna().iloc[0] if f in df.columns and df[f].dropna().shape[0]>0 else None
        if f in numeric_cols:
            input_data[f] = col.number_input(f, value=float(sample_val) if sample_val is not None else 0.0)
        else:
            # categorical -> text
            input_data[f] = col.text_input(f, value=str(sample_val) if sample_val is not None else "")

    if st.button("Predict for these inputs"):
        X_new = pd.DataFrame([input_data])
        # ensure dtypes: numeric conversion for numeric cols
        for c in num_feats:
            if c in X_new.columns:
                X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
        proba = model_pipe.predict_proba(X_new)[0,1]
        st.write(f"Predicted probability of CKD: **{proba:.3f}**")
        # If eGFR present in features or calculate if serum_creatinine present
        egfr = None
        if "eGFR_mdrd" in X_new.columns:
            egfr = X_new.loc[0, "eGFR_mdrd"]
        elif "serum_creatinine" in X_new.columns and "age" in X_new.columns and "female" in X_new.columns:
            egfr = compute_egfr_mdrd(X_new.loc[0,"serum_creatinine"], X_new.loc[0,"age"], bool(int(X_new.loc[0,"female"])))
            st.write(f"Estimated eGFR (MDRD): **{egfr}**")
        if egfr is not None and not pd.isna(egfr):
            st.write("CKD Stage (from eGFR):", stage_from_egfr(egfr))
        else:
            st.write("CKD Stage (from probability):", stage_from_probability(proba))

# -------------------------
# Load existing model
# -------------------------
st.sidebar.title("Utilities")
model_file = st.sidebar.file_uploader("Load trained model (.pkl)", type=["pkl"])
if model_file is not None:
    try:
        loaded = pickle.load(model_file)
        st.session_state["model_pipeline"] = loaded
        st.sidebar.success("Model loaded into session.")
    except Exception as e:
        st.sidebar.error("Failed to load model: " + str(e))

st.sidebar.markdown("---")
st.sidebar.markdown("Need to present? Use the demo dataset, train, and show live predictions.")
st.sidebar.markdown("This is a teaching prototype — not for clinical use.")
