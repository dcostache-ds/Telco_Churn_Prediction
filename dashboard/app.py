from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

st.markdown("""
<style>
    .card-red   { background:#fff0f0; border-left:4px solid #e53935;
                  border-radius:8px; padding:14px 18px; margin-bottom:8px;
                  color:#1a1a1a; }
    .card-green { background:#f0fff0; border-left:4px solid #43a047;
                  border-radius:8px; padding:14px 18px; margin-bottom:8px;
                  color:#1a1a1a; }
    .card-blue  { background:#e8f0fe; border-left:4px solid #1976d2;
                  border-radius:8px; padding:14px 18px; margin-bottom:6px;
                  color:#1a1a1a; }
    .sim-title  { font-size:12px; font-weight:600; color:#333; margin-bottom:4px; }
    .sim-val    { font-size:22px; font-weight:700; color:#1a1a1a; }
    .down       { color:#2e7d32; font-size:13px; font-weight:600; }
    .up         { color:#c62828; font-size:13px; font-weight:600; }
    .same       { color:#555; font-size:13px; }
    .result-lbl { font-size:16px; font-weight:700; color:#1a1a1a; }
    .result-sub { font-size:14px; color:#333; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ── Model ────────────────────────────────────────────────────
@st.cache_resource
def load():
    base = Path(__file__).parent.parent / 'models'
    return (joblib.load(base / 'best_model.pkl'),
            joblib.load(base / 'scaler.pkl'),
            joblib.load(base / 'feature_names.pkl'),
            joblib.load(base / 'best_model_name.pkl'))

model, scaler, feature_names, model_name = load()

# ── Preprocessing manual ─────────────────────────────────────
def predict(client: dict) -> float:
    """
    Construim manual vectorul de features.
    get_dummies pe un singur rand nu functioneaza corect
    pentru ca nu stie toate categoriile posibile.
    """
    row = {col: 0 for col in feature_names}

    # Numerice
    row['SeniorCitizen']  = client['SeniorCitizen']
    row['tenure']         = client['tenure']
    row['MonthlyCharges'] = client['MonthlyCharges']
    row['TotalCharges']   = client['TotalCharges']

    if client['gender'] == 'Male':
        row['gender_Male'] = 1
    if client['Partner'] == 'Yes':
        row['Partner_Yes'] = 1
    if client['Dependents'] == 'Yes':
        row['Dependents_Yes'] = 1
    if client['PhoneService'] == 'Yes':
        row['PhoneService_Yes'] = 1
    if client['MultipleLines'] == 'No phone service':
        row['MultipleLines_No phone service'] = 1
    elif client['MultipleLines'] == 'Yes':
        row['MultipleLines_Yes'] = 1
    if client['InternetService'] == 'Fiber optic':
        row['InternetService_Fiber optic'] = 1
    elif client['InternetService'] == 'No':
        row['InternetService_No'] = 1
    if client['OnlineSecurity'] == 'No internet service':
        row['OnlineSecurity_No internet service'] = 1
    elif client['OnlineSecurity'] == 'Yes':
        row['OnlineSecurity_Yes'] = 1
    if client['OnlineBackup'] == 'No internet service':
        row['OnlineBackup_No internet service'] = 1
    elif client['OnlineBackup'] == 'Yes':
        row['OnlineBackup_Yes'] = 1
    if client['DeviceProtection'] == 'No internet service':
        row['DeviceProtection_No internet service'] = 1
    elif client['DeviceProtection'] == 'Yes':
        row['DeviceProtection_Yes'] = 1
    if client['TechSupport'] == 'No internet service':
        row['TechSupport_No internet service'] = 1
    elif client['TechSupport'] == 'Yes':
        row['TechSupport_Yes'] = 1
    if client['StreamingTV'] == 'No internet service':
        row['StreamingTV_No internet service'] = 1
    elif client['StreamingTV'] == 'Yes':
        row['StreamingTV_Yes'] = 1
    if client['StreamingMovies'] == 'No internet service':
        row['StreamingMovies_No internet service'] = 1
    elif client['StreamingMovies'] == 'Yes':
        row['StreamingMovies_Yes'] = 1
    if client['Contract'] == 'One year':
        row['Contract_One year'] = 1
    elif client['Contract'] == 'Two year':
        row['Contract_Two year'] = 1
    if client['PaperlessBilling'] == 'Yes':
        row['PaperlessBilling_Yes'] = 1
    if client['PaymentMethod'] == 'Credit card (automatic)':
        row['PaymentMethod_Credit card (automatic)'] = 1
    elif client['PaymentMethod'] == 'Electronic check':
        row['PaymentMethod_Electronic check'] = 1
    elif client['PaymentMethod'] == 'Mailed check':
        row['PaymentMethod_Mailed check'] = 1

    df = pd.DataFrame([row])
    num = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num] = scaler.transform(df[num])

    return float(model.predict_proba(df)[0][1]) * 100

def badge(p):
    if p > 70: return "🔴 Risc ridicat"
    if p > 40: return "🟡 Risc mediu"
    return "🟢 Risc scazut"

def delta(orig, nou):
    d = nou - orig
    if d < -0.5: return f'<span class="down">▼ {abs(d):.1f}%</span>'
    if d >  0.5: return f'<span class="up">▲ {d:.1f}%</span>'
    return '<span class="same">≈ fara schimbare</span>'

# ── Header ───────────────────────────────────────────────────
st.title("📊 Churn Prediction")
st.caption(f"Model: **{model_name}**")
st.divider()

# ── Formular ─────────────────────────────────────────────────
with st.expander("📋 Date client", expanded=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Personal**")
        gender     = st.selectbox("Gen", ["Male", "Female"])
        senior     = st.selectbox("Senior Citizen", [0, 1],
                                  format_func=lambda x: "Da" if x else "Nu")
        partner    = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        st.markdown("**Contract & Plata**")
        contract   = st.selectbox("Contract",
                                  ["Month-to-month", "One year", "Two year"])
        paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment    = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        tenure     = st.number_input("Tenure (luni)", 0, 72, 12, step=1)
        monthly    = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=1.0)
        st.number_input("Total Charges ($) — auto",
                        value=float(tenure * monthly), disabled=True)

    with c2:
        st.markdown("**Internet**")
        internet     = st.selectbox("Internet Service",
                                    ["DSL", "Fiber optic", "No"])
        online_sec   = st.selectbox("Online Security",
                                    ["Yes", "No", "No internet service"])
        online_back  = st.selectbox("Online Backup",
                                    ["Yes", "No", "No internet service"])
        device_prot  = st.selectbox("Device Protection",
                                    ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support",
                                    ["Yes", "No", "No internet service"])
        stream_tv    = st.selectbox("Streaming TV",
                                    ["Yes", "No", "No internet service"])
        stream_mov   = st.selectbox("Streaming Movies",
                                    ["Yes", "No", "No internet service"])

    with c3:
        st.markdown("**Telefon**")
        phone    = st.selectbox("Phone Service", ["Yes", "No"])
        m_lines  = st.selectbox("Multiple Lines",
                                ["Yes", "No", "No phone service"])

st.divider()

if st.button("🔮 Prezice Churn", type="primary", use_container_width=True):

    client = {
        'gender': gender, 'SeniorCitizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone,
        'MultipleLines': m_lines, 'InternetService': internet,
        'OnlineSecurity': online_sec, 'OnlineBackup': online_back,
        'DeviceProtection': device_prot, 'TechSupport': tech_support,
        'StreamingTV': stream_tv, 'StreamingMovies': stream_mov,
        'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': tenure * monthly
    }

    try:
        p0 = predict(client)

        # ── Rezultat ─────────────────────────────────────────
        st.markdown("## Rezultat")
        css  = "card-red" if p0 >= 50 else "card-green"
        icon = "⚠️ Clientul probabil va pleca" if p0 >= 50 else "✅ Clientul probabil va ramane"

        st.markdown(f"""
        <div class="{css}">
            <div class="result-lbl">{icon}</div>
            <div class="result-sub">
                Probabilitate churn: <b>{p0:.1f}%</b> &nbsp; {badge(p0)}
            </div>
        </div>""", unsafe_allow_html=True)

        st.progress(p0 / 100)

        # ── What-If ───────────────────────────────────────────
        st.divider()
        st.markdown("## Ce se intampla daca schimbi conditiile?")

        # 1. Contract
        st.markdown("#### Schimbi contractul?")
        cols1 = st.columns(3)
        for i, c_tip in enumerate(["Month-to-month", "One year", "Two year"]):
            sim = client.copy()
            sim['Contract'] = c_tip
            p = predict(sim)
            actual = " ← actual" if c_tip == contract else ""
            css_c  = "card-red" if p >= 50 else "card-green"
            with cols1[i]:
                st.markdown(f"""
                <div class="{css_c}">
                    <div class="sim-title">{c_tip}{actual}</div>
                    <div class="sim-val">{p:.1f}%</div>
                    {delta(p0, p)}
                </div>""", unsafe_allow_html=True)

        # 2. Reducere pret
        st.markdown("#### Oferi o reducere?")
        cols2 = st.columns(4)
        for i, red in enumerate([0, 10, 20, 30]):
            sim    = client.copy()
            new_m  = monthly * (1 - red / 100)
            sim['MonthlyCharges'] = new_m
            sim['TotalCharges']   = tenure * new_m
            p      = predict(sim)
            label  = "Pret actual" if red == 0 else f"-{red}% (${new_m:.0f}/luna)"
            css_c  = "card-red" if p >= 50 else "card-green"
            with cols2[i]:
                st.markdown(f"""
                <div class="{css_c}">
                    <div class="sim-title">{label}</div>
                    <div class="sim-val">{p:.1f}%</div>
                    {delta(p0, p)}
                </div>""", unsafe_allow_html=True)

        # 3. Tenure
        st.markdown("#### Daca clientul sta mai mult?")
        cols3 = st.columns(4)
        for i, extra in enumerate([0, 6, 12, 24]):
            sim    = client.copy()
            new_t  = tenure + extra
            sim['tenure']       = new_t
            sim['TotalCharges'] = new_t * monthly
            p      = predict(sim)
            label  = f"Acum ({tenure}L)" if extra == 0 else f"+{extra}L ({new_t}L)"
            css_c  = "card-red" if p >= 50 else "card-green"
            with cols3[i]:
                st.markdown(f"""
                <div class="{css_c}">
                    <div class="sim-title">{label}</div>
                    <div class="sim-val">{p:.1f}%</div>
                    {delta(p0, p)}
                </div>""", unsafe_allow_html=True)

        # ── Cea mai buna interventie ──────────────────────────
        st.divider()
        st.markdown("## Ce ar ajuta cel mai mult?")

        scenarii = []

        for c_tip in ["One year", "Two year"]:
            if c_tip != contract:
                sim = client.copy()
                sim['Contract'] = c_tip
                scenarii.append((f"Contract {c_tip}", predict(sim)))

        for red in [10, 20, 30]:
            new_m = monthly * (1 - red / 100)
            sim = client.copy()
            sim['MonthlyCharges'] = new_m
            sim['TotalCharges']   = tenure * new_m
            scenarii.append((f"Reducere {red}% la pret (${new_m:.0f}/luna)", predict(sim)))

        for c_tip in ["One year", "Two year"]:
            if c_tip != contract:
                for red in [10, 20]:
                    new_m = monthly * (1 - red / 100)
                    sim = client.copy()
                    sim['Contract'] = c_tip
                    sim['MonthlyCharges'] = new_m
                    sim['TotalCharges']   = tenure * new_m
                    scenarii.append((f"Contract {c_tip} + reducere {red}%", predict(sim)))

        scenarii.sort(key=lambda x: x[1])

        for nume, p in scenarii[:4]:
            reducere = p0 - p
            if reducere > 0.5:
                st.markdown(f"""
                <div class="card-blue">
                    💡 <b style="color:#1a1a1a">{nume}</b><br>
                    <span style="color:#333">
                        {p0:.1f}% → <b>{p:.1f}%</b>
                    </span>
                    &nbsp; <span class="down">▼ -{reducere:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-blue">
                    💡 <b style="color:#1a1a1a">{nume}</b><br>
                    <span style="color:#666; font-size:13px">
                        Impact mic — probabilitate ramane ~{p:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Eroare: {e}")
        st.info("Verifica ca modelele sunt salvate in folderul models/")

st.divider()
st.caption("Customer Churn Prediction — End-to-End ML Project")