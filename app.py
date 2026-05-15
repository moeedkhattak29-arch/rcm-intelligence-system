import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="DenialShield | RCM AI",
    page_icon="🛡️",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4f8ef7;
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #6c757d; }
    .metric-card p  { margin: 4px 0 0; font-size: 22px; font-weight: 600; color: #1a1a2e; }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 18px 0 6px;
    }
    .stAlert > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():
    return {
        "denial": joblib.load("denial_prediction_model.pkl"),
        "reason": joblib.load("denial_reason_model.pkl"),
        "reason_le": joblib.load("denial_reason_label_encoder.pkl"),
        "appeal": joblib.load("appeal_success_model.pkl"),
        "ar": joblib.load("ar_days_model.pkl"),
    }

try:
    models = load_models()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ==================================================
# MODEL METRICS (pre-computed on test set)
# ==================================================
MODEL_METRICS = {
    "denial": {
        "accuracy": 0.77, "f1": 0.85, "auc": 0.71,
        "cm": [[24, 37], [9, 130]],
        "top_features": [
            ("Days in AR", 0.2634), ("Reason Code: Unknown", 0.1365),
            ("CPT Code", 0.1176), ("AR Status: Active", 0.0269),
            ("AR Status: Resolved", 0.0269)
        ],
        "labels": ["Paid", "Denied"]
    },
    "appeal": {
        "accuracy": 0.59, "f1": 0.59, "auc": 0.61,
        "cm": [[58, 41], [41, 60]],
        "top_features": [
            ("AR Days", 0.2385), ("Patient Age", 0.198),
            ("Prior Appeals", 0.0892), ("Denial: Lack of Info", 0.0392),
            ("Denial: Timely Filing", 0.0331)
        ],
        "labels": ["Failed", "Success"]
    },
    "ar": {
        "mae": 3.40, "rmse": 4.31, "r2": 0.982,
        "top_features": [
            ("Resubmission Count", 0.494), ("Billing Lag", 0.2846),
            ("Denial History", 0.1612), ("Claim Amount", 0.0466),
            ("Patient Responsibility", 0.0056)
        ]
    }
}

# ==================================================
# DROPDOWN VALUES (from actual training data)
# ==================================================
INSURANCE_TYPES   = ["BCBS", "Medicaid", "Medicare", "Private"]
AR_STATUS_OPTS    = ["Active", "Pending", "Resolved"]
REASON_CODES      = ["CO-16", "CO-18", "CO-50", "CO-97", "PR-2", "PR-27", "Unknown"]
DIAGNOSIS_CODES   = ["E11.9", "F32.9", "I10", "J45.909", "L57.0", "L97.9", "M17.11", "Z00.00"]
CPT_CODES_RCM     = [17000, 20610, 90837, 99204, 99213, 99214]

PAYERS_FULL       = ["Aetna", "BCBS", "Cigna", "Medicare", "United Healthcare"]
ICD10_CODES       = ["E11.9", "I10", "J06.9", "M54.50", "Z00.00"]
MODIFIERS         = ["None", "25", "59", "LT", "RT"]

APPEAL_DENIAL_REASONS = [
    "Coordination of Benefits", "Duplicate Claim",
    "Experimental/Investigational", "Incorrect Coding",
    "Lack of Information", "Medical Necessity",
    "Non-Covered Service", "Prior Authorization Missing",
    "Provider Not in Network", "Timely Filing"
]

PAYERS_HC = ["Aetna", "BlueShield", "Cigna", "Humana", "Medicare", "State Medicaid", "UnitedHealth"]
CPT_CODES_HC = [11102, 36415, 90658, 93000, 99213, 99214]

# ==================================================
# HELPER: METRIC CARDS
# ==================================================
def metric_card(label, value, color="#4f8ef7"):
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <h3>{label}</h3>
        <p>{value}</p>
    </div>""", unsafe_allow_html=True)

# ==================================================
# HELPER: CONFUSION MATRIX PLOT
# ==================================================
def plot_confusion_matrix(cm, labels):
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred: {l}" for l in labels], y=[f"Actual: {l}" for l in labels],
        text=cm, texttemplate="%{text}",
        colorscale="Blues", showscale=False
    ))
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20), height=260,
        font=dict(size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ==================================================
# HELPER: FEATURE IMPORTANCE PLOT
# ==================================================
def plot_feature_importance(top_features):
    names = [f[0] for f in top_features]
    vals  = [f[1] for f in top_features]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker_color="#4f8ef7", text=[f"{v:.3f}" for v in vals],
        textposition="outside"
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=60), height=220,
        xaxis_title="Importance", yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    return fig

# ==================================================
# HELPER: PROBABILITY GAUGE
# ==================================================
def plot_gauge(prob, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%"},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#e8f5e9"},
                {"range": [40, 70], "color": "#fff8e1"},
                {"range": [70, 100], "color": "#fce4ec"},
            ]
        }
    ))
    fig.update_layout(margin=dict(t=40, b=10, l=20, r=20), height=200,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==================================================
# HELPER: MODEL PERFORMANCE EXPANDER
# ==================================================
def show_model_performance(key, is_regression=False):
    m = MODEL_METRICS[key]
    with st.expander("📊 Model Performance Metrics", expanded=False):
        if not is_regression:
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Accuracy", f"{m['accuracy']:.0%}", "#4f8ef7")
            with c2: metric_card("F1 Score", f"{m['f1']:.0%}", "#28a745")
            with c3: metric_card("AUC-ROC", f"{m['auc']:.0%}", "#fd7e14")

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
                st.plotly_chart(plot_confusion_matrix(m["cm"], m["labels"]),
                                use_container_width=True, config={"displayModeBar": False})
            with cc2:
                st.markdown('<p class="section-header">Top Feature Importances</p>', unsafe_allow_html=True)
                st.plotly_chart(plot_feature_importance(m["top_features"]),
                                use_container_width=True, config={"displayModeBar": False})
        else:
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("MAE", f"{m['mae']} days", "#4f8ef7")
            with c2: metric_card("RMSE", f"{m['rmse']} days", "#fd7e14")
            with c3: metric_card("R² Score", f"{m['r2']:.3f}", "#28a745")
            st.markdown('<p class="section-header">Top Feature Importances</p>', unsafe_allow_html=True)
            st.plotly_chart(plot_feature_importance(m["top_features"]),
                            use_container_width=True, config={"displayModeBar": False})

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=56)
st.sidebar.title("DenialShield")
st.sidebar.caption("Revenue Cycle Management · AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🧾 Denial Prediction", "🔎 Denial Reason",
     "📈 Appeal Success", "⏳ AR Days Forecast", "📂 Bulk Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("Built for Medical Billing & RCM Analytics\n\n**4 ML Models · 4 Datasets**")

# ==================================================
# PAGE 0 — OVERVIEW / EDA DASHBOARD
# ==================================================
if page == "🏠 Overview":
    st.title("🛡️ DenialShield — RCM Intelligence Platform")
    st.markdown("AI-powered claims denial prediction, root-cause analysis, and AR forecasting for healthcare billing teams.")
    st.markdown("---")

    # Load data for EDA
    df_rcm   = pd.read_csv("rcm_claims_data__1_.csv")
    df_full  = pd.read_csv("rcm_full_dataset__1_.csv")
    df_appeal= pd.read_csv("Appeal_dataset.csv")
    df_hc    = pd.read_csv("healthcare_claims.csv")
    df_rcm['Reason Code'] = df_rcm['Reason Code'].fillna('Unknown')

    # KPI row
    total    = len(df_rcm)
    denied   = df_rcm['denial'].sum()
    denial_r = denied / total
    appeal_r = df_appeal['success'].mean()
    avg_ar   = df_hc['AR_Days'].mean()

    st.subheader("📊 Key Metrics at a Glance")
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Total Claims", f"{total:,}", "#4f8ef7")
    with k2: metric_card("Denial Rate", f"{denial_r:.1%}", "#dc3545")
    with k3: metric_card("Appeal Win Rate", f"{appeal_r:.1%}", "#28a745")
    with k4: metric_card("Avg AR Days", f"{avg_ar:.0f} days", "#fd7e14")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Denial Rate by Insurance Type")
        denial_by_ins = df_rcm.groupby('Insurance Type')['denial'].mean().reset_index()
        denial_by_ins.columns = ['Insurance Type', 'Denial Rate']
        fig = px.bar(denial_by_ins, x='Insurance Type', y='Denial Rate',
                     color='Denial Rate', color_continuous_scale='Reds',
                     text=denial_by_ins['Denial Rate'].apply(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_tickformat='.0%',
                          paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Denial Reason Codes")
        rc_counts = df_rcm['Reason Code'].value_counts().reset_index()
        rc_counts.columns = ['Reason Code', 'Count']
        fig2 = px.pie(rc_counts, names='Reason Code', values='Count',
                      color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("AR Days Distribution")
        fig3 = px.histogram(df_hc, x='AR_Days', nbins=30,
                            color_discrete_sequence=["#4f8ef7"])
        fig3.add_vline(x=df_hc['AR_Days'].mean(), line_dash="dash",
                       line_color="red", annotation_text=f"Mean: {avg_ar:.0f}d")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                           xaxis_title="AR Days", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Appeal Success by Denial Reason")
        appeal_by_reason = df_appeal.groupby('denial_reason')['success'].mean().sort_values().reset_index()
        fig4 = px.bar(appeal_by_reason, x='success', y='denial_reason',
                      orientation='h', color='success',
                      color_continuous_scale='Greens',
                      text=appeal_by_reason['success'].apply(lambda x: f"{x:.0%}"))
        fig4.update_traces(textposition='outside')
        fig4.update_layout(showlegend=False, xaxis_tickformat='.0%',
                           paper_bgcolor="rgba(0,0,0,0)", height=320,
                           xaxis_title="Win Rate", yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Models in This App")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("**🧾 Denial Prediction**")
        st.markdown("Random Forest · Binary Classification")
        st.markdown(f"AUC-ROC: **71%** · F1: **85%**")
    with m2:
        st.markdown("**🔎 Denial Reason**")
        st.markdown("Random Forest · Multi-class Classification")
        st.markdown(f"Accuracy: **95%** · F1: **95%**")
    with m3:
        st.markdown("**📈 Appeal Success**")
        st.markdown("Random Forest · Binary Classification")
        st.markdown(f"AUC-ROC: **61%** · F1: **59%**")
    with m4:
        st.markdown("**⏳ AR Days Forecast**")
        st.markdown("Random Forest · Regression")
        st.markdown(f"R²: **0.982** · MAE: **3.4 days**")

# ==================================================
# PAGE 1 — DENIAL PREDICTION
# ==================================================
elif page == "🧾 Denial Prediction":
    st.title("🧾 Claim Denial Prediction")
    st.write("Predict whether a claim will be **PAID** or **DENIED** before submission.")
    show_model_performance("denial")
    st.markdown("---")

    with st.form("denial_form"):
        st.subheader("Enter Claim Details")
        col1, col2 = st.columns(2)
        with col1:
            cpt = st.selectbox("CPT Code", CPT_CODES_RCM)
            diag = st.selectbox("Diagnosis Code", DIAGNOSIS_CODES)
            insurance = st.selectbox("Insurance Type", INSURANCE_TYPES)
        with col2:
            ar_status = st.selectbox("AR Status", AR_STATUS_OPTS)
            reason_code = st.selectbox("Reason Code", REASON_CODES)
            days_in_ar = st.number_input("Days in AR", min_value=0, max_value=180, value=30)

        submitted = st.form_submit_button("🔍 Predict Denial Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "CPT Code": cpt, "Diagnosis Code": diag,
            "Insurance Type": insurance, "AR Status": ar_status,
            "Reason Code": reason_code, "Days in AR": days_in_ar
        }])

        pred = models["denial"].predict(input_df)[0]
        prob = models["denial"].predict_proba(input_df)[0][1]
        denial_prob = prob
        paid_prob   = 1 - prob

        st.markdown("---")
        st.subheader("Prediction Result")

        g1, g2 = st.columns(2)
        with g1:
            color = "#dc3545" if pred == 1 else "#28a745"
            st.plotly_chart(plot_gauge(denial_prob, "Denial Probability", color),
                            use_container_width=True, config={"displayModeBar": False})
        with g2:
            if pred == 1:
                st.error(f"### ❌ High Risk of Denial")
                st.markdown(f"**Denial probability:** `{denial_prob:.1%}`")
                st.markdown("**Recommended actions:**")
                st.markdown("- Verify diagnosis and CPT code pairing")
                st.markdown("- Check prior authorization requirements")
                st.markdown("- Review reason code documentation")
                st.markdown("- Confirm patient eligibility before submission")
            else:
                st.success(f"### ✅ Likely to be Paid")
                st.markdown(f"**Paid probability:** `{paid_prob:.1%}`")
                st.markdown("**Tips to maintain approval:**")
                st.markdown("- Submit claim promptly to avoid timely filing")
                st.markdown("- Ensure complete documentation is attached")
                st.markdown("- Double-check modifier codes if applicable")

# ==================================================
# PAGE 2 — DENIAL REASON
# ==================================================
elif page == "🔎 Denial Reason":
    st.title("🔎 Denial Reason Analysis")
    st.write("Predict the **primary denial reason code** for a claim — before it is rejected.")

    with st.expander("📊 Model Performance Metrics", expanded=False):
        c1, c2 = st.columns(2)
        with c1: metric_card("Accuracy", "94.7%", "#4f8ef7")
        with c2: metric_card("F1 Score (weighted)", "94.7%", "#28a745")
        st.info("ℹ️ This model only applies to claims flagged as denied. It predicts among CO-16, CO-18, CO-50, CO-97, CO-197, PR-1 codes.")

    st.markdown("---")

    with st.form("reason_form"):
        st.subheader("Enter Claim Details")
        col1, col2 = st.columns(2)
        with col1:
            payer = st.selectbox("Payer", PAYERS_FULL)
            cpt   = st.number_input("CPT Code", min_value=10000, max_value=99999, value=99213)
            icd10 = st.selectbox("ICD-10 Code", ICD10_CODES)
            modifier = st.selectbox("Modifier", MODIFIERS)
        with col2:
            amount    = st.number_input("Claim Amount ($)", min_value=0.0, value=250.0)
            age       = st.number_input("Patient Age", 0, 120, 45)
            prior_auth= st.selectbox("Prior Authorization", ["No (0)", "Yes (1)"])
            doc_score = st.slider("Documentation Quality Score", 0.0, 1.0, 0.8, 0.05)

        submitted = st.form_submit_button("🧠 Analyze Denial Reason", use_container_width=True)

    if submitted:
        prior_auth_val = 1 if "Yes" in prior_auth else 0
        input_df = pd.DataFrame([{
            "payer": payer, "cpt": cpt, "icd10": icd10,
            "modifier": modifier if modifier != "None" else np.nan,
            "amount": amount, "age": age,
            "prior_auth": prior_auth_val, "doc_score": doc_score
        }])

        pred_idx = models["reason"].predict(input_df)[0]
        reason   = models["reason_le"].inverse_transform([pred_idx])[0]
        proba    = models["reason"].predict_proba(input_df)[0]
        classes  = models["reason_le"].classes_

        REASON_DESCRIPTIONS = {
            "CO-16":  "Claim lacks information — missing data or documentation",
            "CO-18":  "Duplicate claim — already submitted or adjudicated",
            "CO-50":  "These services are not covered under the patient's plan",
            "CO-97":  "Payment was adjusted because it was not covered",
            "CO-197": "Precertification / authorization / notification absent",
            "PR-1":   "Deductible amount — patient responsibility",
        }

        st.markdown("---")
        st.subheader("Prediction Result")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.warning(f"### ⚠️ {reason}")
            st.markdown(f"*{REASON_DESCRIPTIONS.get(reason, 'See payer guidelines')}*")

        with r2:
            st.markdown("**Probability across all denial codes:**")
            prob_df = pd.DataFrame({"Code": classes, "Probability": proba}).sort_values("Probability", ascending=True)
            fig = go.Figure(go.Bar(
                x=prob_df["Probability"], y=prob_df["Code"],
                orientation='h', marker_color="#4f8ef7",
                text=[f"{v:.1%}" for v in prob_df["Probability"]],
                textposition="outside"
            ))
            fig.update_layout(height=240, margin=dict(t=10,b=10,l=10,r=60),
                              xaxis_tickformat='.0%', paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.info(f"💡 **Action:** {REASON_DESCRIPTIONS.get(reason, '')} — address this before submission.")

# ==================================================
# PAGE 3 — APPEAL SUCCESS
# ==================================================
elif page == "📈 Appeal Success":
    st.title("📈 Appeal Success Prediction")
    st.write("Estimate the **likelihood of winning a denied claim appeal**.")
    show_model_performance("appeal")
    st.markdown("---")

    with st.form("appeal_form"):
        st.subheader("Enter Appeal Details")
        col1, col2 = st.columns(2)
        with col1:
            patient_age    = st.number_input("Patient Age", 0, 120, 45)
            gender         = st.selectbox("Gender", ["Female", "Male", "Other"])
            claim_type     = st.selectbox("Claim Type", ["Emergency", "Inpatient", "Outpatient"])
            insurance_type = st.selectbox("Insurance Type", ["Medicaid", "Medicare", "Private", "Workers Comp"])
        with col2:
            denial_reason  = st.selectbox("Denial Reason", APPEAL_DENIAL_REASONS)
            ar_days        = st.number_input("Days in AR", min_value=0, max_value=200, value=30)
            prior_appeals  = st.number_input("Prior Appeals Count", min_value=0, max_value=10, value=0)

        submitted = st.form_submit_button("📊 Predict Appeal Outcome", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "patient_age": patient_age, "gender": gender,
            "claim_type": claim_type, "insurance_type": insurance_type,
            "denial_reason": denial_reason, "ar_days": ar_days,
            "prior_appeals": prior_appeals
        }])

        pred = models["appeal"].predict(input_df)[0]
        prob = models["appeal"].predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")
        g1, g2 = st.columns(2)
        with g1:
            color = "#28a745" if pred == 1 else "#dc3545"
            st.plotly_chart(plot_gauge(prob, "Appeal Success Probability", color),
                            use_container_width=True, config={"displayModeBar": False})
        with g2:
            if pred == 1:
                st.success(f"### ✅ Appeal Likely to Succeed")
                st.markdown(f"**Success probability:** `{prob:.1%}`")
                st.markdown("**To strengthen your appeal:**")
                st.markdown("- Submit supporting clinical documentation")
                st.markdown("- Include a letter of medical necessity if applicable")
                st.markdown("- Reference payer policy page that covers this service")
            else:
                st.error(f"### ❌ Appeal Likely to Fail")
                st.markdown(f"**Failure probability:** `{(1-prob):.1%}`")
                st.markdown("**Consider these alternatives:**")
                st.markdown("- Request a peer-to-peer review with the payer's medical director")
                st.markdown("- Recode the claim if a coding error was the denial reason")
                st.markdown("- Check if secondary insurance can cover the balance")
                st.markdown("- Escalate to a external independent review organization (IRO)")

# ==================================================
# PAGE 4 — AR DAYS FORECAST
# ==================================================
elif page == "⏳ AR Days Forecast":
    st.title("⏳ AR Days Forecast")
    st.write("Predict **how long a claim will remain in Accounts Receivable** before resolution.")
    show_model_performance("ar", is_regression=True)
    st.markdown("---")

    with st.form("ar_form"):
        st.subheader("Enter Claim Details")
        col1, col2 = st.columns(2)
        with col1:
            payer                = st.selectbox("Payer", PAYERS_HC)
            cpt_code             = st.selectbox("CPT Code", CPT_CODES_HC)
            claim_amount         = st.number_input("Claim Amount ($)", 0.0, 10000.0, 500.0)
            denial_history       = st.selectbox("Prior Denial History Count", [0, 1, 2])
        with col2:
            resubmission_count   = st.number_input("Resubmission Count", 0, 10, 0)
            patient_responsibility = st.number_input("Patient Responsibility ($)", 0.0, 5000.0, 50.0)
            billing_lag          = st.number_input("Billing Lag (Days)", 0, 60, 5)

        submitted = st.form_submit_button("⏱️ Estimate AR Days", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Payer": payer, "CPT_Code": cpt_code,
            "Claim_Amount": claim_amount, "Denial_History": denial_history,
            "Resubmission_Count": resubmission_count,
            "Patient_Responsibility": patient_responsibility,
            "Billing_Lag": billing_lag
        }])

        days = float(models["ar"].predict(input_df)[0])
        days = round(days, 1)

        st.markdown("---")
        st.subheader("Forecast Result")
        r1, r2, r3 = st.columns(3)
        with r1: metric_card("Estimated AR Duration", f"{days:.0f} days", "#4f8ef7")
        with r2: metric_card("Resolution Date (est.)", f"~{int(days)} days from today", "#fd7e14")
        with r3:
            risk = "🔴 High Risk" if days > 60 else ("🟡 Medium Risk" if days > 30 else "🟢 Low Risk")
            risk_color = "#dc3545" if days > 60 else ("#fd7e14" if days > 30 else "#28a745")
            metric_card("AR Risk Level", risk, risk_color)

        # AR timeline visual
        benchmarks = {"Industry avg": 45, "Medicare std": 30, "Your claim": days}
        fig = go.Figure()
        colors = {"Industry avg": "#adb5bd", "Medicare std": "#74c0fc", "Your claim": "#4f8ef7"}
        for label, val in benchmarks.items():
            fig.add_trace(go.Bar(name=label, x=[label], y=[val],
                                 marker_color=colors[label],
                                 text=[f"{val:.0f}d"], textposition="outside"))
        fig.update_layout(
            title="Your Claim vs. Industry Benchmarks",
            yaxis_title="Days in AR", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if days > 60:
            st.warning("⚠️ **High AR Risk** — Immediate action recommended: initiate payer follow-up call, escalate to senior billing staff, consider appeal if previously denied.")
        elif days > 30:
            st.info("🟡 **Moderate AR Risk** — Schedule a follow-up within 2 weeks. Verify claim status on payer portal.")
        else:
            st.success("✅ **Low AR Risk** — Claim is on track. Standard monitoring applies.")

# ==================================================
# PAGE 5 — BULK PREDICTION
# ==================================================
elif page == "📂 Bulk Prediction":
    st.title("📂 Bulk Claim Prediction")
    st.write("Upload a CSV of claims and get denial predictions for all rows at once.")
    st.markdown("---")

    st.subheader("Required CSV Format")
    sample = pd.DataFrame([{
        "CPT Code": 99213, "Diagnosis Code": "Z00.00",
        "Insurance Type": "Medicare", "AR Status": "Pending",
        "Reason Code": "CO-16", "Days in AR": 20
    }, {
        "CPT Code": 90837, "Diagnosis Code": "F32.9",
        "Insurance Type": "BCBS", "AR Status": "Active",
        "Reason Code": "CO-18", "Days in AR": 45
    }])
    st.dataframe(sample, use_container_width=True)

    # Download sample
    csv_bytes = sample.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample Template", csv_bytes,
                       "sample_claims.csv", "text/csv")

    st.markdown("---")
    uploaded = st.file_uploader("Upload your claims CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_upload):,} claims")
        st.dataframe(df_upload.head(5), use_container_width=True)

        required_cols = ["CPT Code", "Diagnosis Code", "Insurance Type",
                         "AR Status", "Reason Code", "Days in AR"]
        missing_cols  = [c for c in required_cols if c not in df_upload.columns]

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            if st.button("🚀 Run Bulk Prediction", use_container_width=True):
                with st.spinner("Running predictions..."):
                    df_upload['Reason Code'] = df_upload['Reason Code'].fillna('Unknown')
                    preds = models["denial"].predict(df_upload[required_cols])
                    probs = models["denial"].predict_proba(df_upload[required_cols])[:,1]

                    df_upload['Denial Prediction'] = np.where(preds == 1, "DENIED", "PAID")
                    df_upload['Denial Probability'] = (probs * 100).round(1)
                    df_upload['Risk Level'] = pd.cut(
                        probs, bins=[0, 0.4, 0.7, 1.0],
                        labels=["Low Risk", "Medium Risk", "High Risk"]
                    )

                st.subheader("Results")
                st.dataframe(df_upload, use_container_width=True)

                # Summary stats
                denied_n = (preds == 1).sum()
                c1, c2, c3 = st.columns(3)
                with c1: metric_card("Total Claims", f"{len(df_upload):,}")
                with c2: metric_card("Predicted Denied", f"{denied_n:,}", "#dc3545")
                with c3: metric_card("Denial Rate", f"{denied_n/len(df_upload):.1%}", "#fd7e14")

                # Download results
                out_csv = df_upload.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results CSV", out_csv,
                                   "denial_predictions.csv", "text/csv",
                                   use_container_width=True)
