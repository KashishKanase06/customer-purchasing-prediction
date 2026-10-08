"""
frontend.py
===========
Streamlit frontend for the Customer Purchase Prediction project.

Pages:
  1. 📊 Dataset Overview   – shape, sample, class distribution
  2. 🔍 Exploratory Analysis – charts for key features
  3. 🤖 Model Performance   – accuracy, ROC-AUC, confusion matrix, feature importance
  4. 🛒 Live Prediction      – interactive form → POST /predict → result

Run (make sure the FastAPI backend is running on port 8000 first):
    streamlit run frontend.py
"""

import os
import warnings
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
DATA_PATH  = os.path.join("data", "online_shoppers_intention.csv")
MODELS_DIR = "models"

VALID_MONTHS  = ["Jan", "Feb", "Mar", "Apr", "May", "June",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VALID_VISITOR = ["Returning_Visitor", "New_Visitor", "Other"]

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Weekend"] = df["Weekend"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    df["Revenue"]  = df["Revenue"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
    return df

@st.cache_resource
def load_artifacts():
    try:
        model        = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
        scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        encoders     = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        metadata     = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
        return model, scaler, encoders, feature_cols, metadata
    except FileNotFoundError:
        return None, None, None, None, None

def get_test_data(df, encoders, feature_cols):
    """Reproduce the exact train/test split used during training."""
    dfc = df.copy()
    for col in ["Month", "VisitorType"]:
        dfc[col] = encoders[col].transform(dfc[col].astype(str))
    X = dfc[feature_cols]
    y = dfc["Revenue"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shopping-cart.png",
    width=80,
)
st.sidebar.title("🛒 Purchase Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dataset Overview",
     "🔍 Exploratory Analysis",
     "🤖 Model Performance",
     "🛒 Live Prediction"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Stack**\n"
    "- 🐍 Python\n"
    "- 🌲 Random Forest\n"
    "- ⚡ FastAPI\n"
    "- 🎈 Streamlit\n"
    "- 📊 Plotly"
)

# ── Load data & artifacts ─────────────────────────────────────────────────────
df = load_data()
model, scaler, encoders, feature_cols, metadata = load_artifacts()
artifacts_ready = model is not None

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown(
        "The **Online Shoppers Intention** dataset contains "
        f"**{len(df):,}** browsing sessions with **{len(df.columns)}** features. "
        "The target column `Revenue` indicates whether the session led to a purchase."
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions",    f"{len(df):,}")
    c2.metric("Purchased",         f"{df['Revenue'].sum():,}")
    c3.metric("Not Purchased",     f"{(df['Revenue'] == 0).sum():,}")
    c4.metric("Purchase Rate",     f"{df['Revenue'].mean()*100:.1f}%")

    st.markdown("---")

    # Target distribution donut
    col_a, col_b = st.columns([1, 2])
    with col_a:
        fig_donut = px.pie(
            names=["No Purchase", "Purchase"],
            values=[
                (df["Revenue"] == 0).sum(),
                df["Revenue"].sum()
            ],
            hole=0.55,
            color_discrete_sequence=["#e5e7eb", "#3b82d4"],
            title="Target Distribution",
        )
        fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.subheader("Sample Data (first 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[num_cols].describe().T.style.format("{:.3f}"), use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("✅ No missing values in the dataset.")
    else:
        st.dataframe(missing.rename("Missing Count"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Exploratory Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    # Monthly purchase rate
    st.subheader("Purchase Rate by Month")
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_stats = (
        df.groupby("Month")["Revenue"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Purchase Rate", "count": "Sessions"})
        .reset_index()
    )
    month_stats["Purchase Rate %"] = (month_stats["Purchase Rate"] * 100).round(2)
    try:
        month_stats["Month"] = pd.Categorical(month_stats["Month"], categories=month_order, ordered=True)
        month_stats = month_stats.sort_values("Month")
    except Exception:
        pass

    fig_month = px.bar(
        month_stats, x="Month", y="Purchase Rate %",
        color="Purchase Rate %",
        color_continuous_scale="Blues",
        text="Purchase Rate %",
        title="Purchase Rate (%) per Month",
    )
    fig_month.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_month.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_month, use_container_width=True)

    # Visitor type
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Visitor Type Distribution")
        vt = df["VisitorType"].value_counts().reset_index()
        vt.columns = ["VisitorType", "Count"]
        fig_vt = px.bar(vt, x="VisitorType", y="Count",
                        color="VisitorType",
                        color_discrete_sequence=["#3b82d4", "#7c5cd8", "#57606a"])
        fig_vt.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_vt, use_container_width=True)

    with col2:
        st.subheader("Purchase Rate by Visitor Type")
        vt_rev = (
            df.groupby("VisitorType")["Revenue"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
            .rename(columns={"Revenue": "Purchase Rate %"})
        )
        fig_vtr = px.bar(
            vt_rev, x="VisitorType", y="Purchase Rate %",
            color="Purchase Rate %",
            color_continuous_scale="Blues",
            text="Purchase Rate %",
        )
        fig_vtr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_vtr.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_vtr, use_container_width=True)

    # Weekend effect
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Weekend vs Weekday Purchase Rate")
        wd = (
            df.groupby("Weekend")["Revenue"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
        )
        wd["Weekend"] = wd["Weekend"].map({1: "Weekend", 0: "Weekday"})
        wd.columns = ["Day Type", "Purchase Rate %"]
        fig_wd = px.bar(wd, x="Day Type", y="Purchase Rate %",
                        color="Day Type",
                        color_discrete_sequence=["#3b82d4", "#7c5cd8"],
                        text="Purchase Rate %")
        fig_wd.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_wd.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_wd, use_container_width=True)

    with col4:
        st.subheader("Page Values Distribution (log)")
        df_nz = df[df["PageValues"] > 0]
        fig_pv = px.histogram(
            df_nz, x="PageValues", color="Revenue",
            nbins=60, log_y=True,
            barmode="overlay",
            color_discrete_map={0: "#e5e7eb", 1: "#3b82d4"},
            labels={"Revenue": "Purchased"},
            title="PageValues (non-zero sessions)",
        )
        fig_pv.update_layout(height=320)
        st.plotly_chart(fig_pv, use_container_width=True)

    # Bounce vs Exit scatter
    st.subheader("Bounce Rate vs Exit Rate (by Purchase Outcome)")
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig_sc = px.scatter(
        sample,
        x="BounceRates", y="ExitRates",
        color=sample["Revenue"].map({1: "Purchase", 0: "No Purchase"}),
        opacity=0.5,
        color_discrete_map={"Purchase": "#3b82d4", "No Purchase": "#e5e7eb"},
        title="Bounce Rate vs Exit Rate",
    )
    fig_sc.update_layout(height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation Matrix",
        height=600,
    )
    fig_heat.update_layout(margin=dict(t=60, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    if not artifacts_ready:
        st.warning("⚠️ Model artifacts not found. Please run `python train_model.py` first.")
        st.stop()

    # Metadata KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",    metadata["model_name"])
    c2.metric("Accuracy",      f"{metadata['accuracy']*100:.2f}%")
    c3.metric("ROC-AUC",       f"{metadata['roc_auc']:.4f}")
    c4.metric("Training Size", f"{metadata['train_size']:,}")

    st.markdown("---")

    # Rebuild test set for charts
    X_test, y_test = get_test_data(df, encoders, feature_cols)

    model_name = metadata.get("model_name", "")
    if model_name == "LogisticRegression":
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    y_pred  = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]

    # Confusion matrix
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Purchase", "Purchase"],
            y=["No Purchase", "Purchase"],
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    with col_b:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {roc_auc_val:.4f}",
            line=dict(color="#3b82d4", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#e5e7eb", dash="dash"),
            showlegend=False,
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curve",
            height=380,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # Feature importances
    if hasattr(model, "feature_importances_"):
        st.subheader("Top Feature Importances")
        fi = pd.Series(model.feature_importances_, index=feature_cols)
        fi_sorted = fi.nlargest(15).reset_index()
        fi_sorted.columns = ["Feature", "Importance"]
        fig_fi = px.bar(
            fi_sorted, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Top 15 Feature Importances",
        )
        fig_fi.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            height=480,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Live Prediction":
    st.title("🛒 Live Purchase Prediction")
    st.markdown(
        "Fill in the browsing session details below and click **Predict** "
        "to get an instant purchase probability."
    )

    if not artifacts_ready:
        st.warning("⚠️ Model artifacts not found. Please run `python train_model.py` first.")

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("📋 Session Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Page Counts**")
            administrative   = st.number_input("Administrative Pages",   0, 100, 0)
            informational    = st.number_input("Informational Pages",    0, 100, 0)
            product_related  = st.number_input("Product Related Pages",  0, 500, 5)

        with col2:
            st.markdown("**Time Spent (seconds)**")
            admin_duration   = st.number_input("Admin Duration",   0.0, 10000.0, 0.0, step=10.0)
            info_duration    = st.number_input("Info Duration",    0.0, 10000.0, 0.0, step=10.0)
            product_duration = st.number_input("Product Duration", 0.0, 50000.0, 200.0, step=10.0)

        with col3:
            st.markdown("**Engagement Metrics**")
            bounce_rates = st.slider("Bounce Rate",  0.0, 1.0, 0.02, step=0.01)
            exit_rates   = st.slider("Exit Rate",    0.0, 1.0, 0.05, step=0.01)
            page_values  = st.number_input("Page Values", 0.0, 400.0, 10.0, step=1.0)
            special_day  = st.slider("Special Day Proximity", 0.0, 1.0, 0.0, step=0.2)

        st.markdown("---")
        col4, col5 = st.columns(2)

        with col4:
            st.markdown("**Session Context**")
            month        = st.selectbox("Month",        VALID_MONTHS, index=9)   # Nov
            visitor_type = st.selectbox("Visitor Type", VALID_VISITOR)
            weekend      = st.radio("Weekend?", ["No", "Yes"], horizontal=True)

        with col5:
            st.markdown("**Technical Details**")
            os_type      = st.selectbox("Operating System", list(range(1, 9)), index=1)
            browser      = st.selectbox("Browser",          list(range(1, 14)), index=1)
            region       = st.selectbox("Region",           list(range(1, 10)), index=0)
            traffic_type = st.selectbox("Traffic Type",     list(range(1, 21)), index=2)

        st.markdown("")
        submitted = st.form_submit_button("🔮 Predict Purchase Intent", use_container_width=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    if submitted:
        weekend_int = 1 if weekend == "Yes" else 0

        payload = {
            "Administrative":           administrative,
            "Administrative_Duration":  admin_duration,
            "Informational":            informational,
            "Informational_Duration":   info_duration,
            "ProductRelated":           product_related,
            "ProductRelated_Duration":  product_duration,
            "BounceRates":              bounce_rates,
            "ExitRates":                exit_rates,
            "PageValues":               page_values,
            "SpecialDay":               special_day,
            "Month":                    month,
            "OperatingSystems":         os_type,
            "Browser":                  browser,
            "Region":                   region,
            "TrafficType":              traffic_type,
            "VisitorType":              visitor_type,
            "Weekend":                  weekend_int,
        }

        # Try the FastAPI backend first; fall back to direct model inference
        result = None
        source = "API"

        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
        except Exception:
            source = "Local"
            if artifacts_ready:
                try:
                    month_enc   = encoders["Month"].transform([month])[0]
                    visitor_enc = encoders["VisitorType"].transform([visitor_type])[0]
                    row = np.array([[
                        administrative, admin_duration,
                        informational,  info_duration,
                        product_related, product_duration,
                        bounce_rates, exit_rates, page_values, special_day,
                        month_enc, os_type, browser, region,
                        traffic_type, visitor_enc, weekend_int,
                    ]])
                    model_name = metadata.get("model_name", "")
                    row_input  = scaler.transform(row) if model_name == "LogisticRegression" else row
                    pred  = int(model.predict(row_input)[0])
                    proba = model.predict_proba(row_input)[0]
                    result = {
                        "will_purchase":    bool(pred),
                        "purchase_label":   "Will Purchase ✓" if pred else "Will NOT Purchase ✗",
                        "confidence":       round(float(max(proba)) * 100, 2),
                        "purchase_prob":    round(float(proba[1]) * 100, 2),
                        "no_purchase_prob": round(float(proba[0]) * 100, 2),
                    }
                except Exception as e:
                    st.error(f"Local inference failed: {e}")

        if result:
            st.markdown("---")
            st.subheader(f"🎯 Prediction Result  *(via {source})*")

            if result["will_purchase"]:
                st.success(f"### ✅ {result['purchase_label']}")
            else:
                st.error(f"### ❌ {result['purchase_label']}")

            r1, r2, r3 = st.columns(3)
            r1.metric("Purchase Probability",    f"{result['purchase_prob']:.1f}%")
            r2.metric("No-Purchase Probability", f"{result['no_purchase_prob']:.1f}%")
            r3.metric("Model Confidence",        f"{result['confidence']:.1f}%")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result["purchase_prob"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Purchase Probability (%)"},
                delta={"reference": 15.5},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#3b82d4"},
                    "steps": [
                        {"range": [0,  30], "color": "#fee2e2"},
                        {"range": [30, 60], "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#1f2328", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("📤 View raw payload sent"):
                st.json(payload)
