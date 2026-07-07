import streamlit as st
import joblib
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0B1220 0%, #10192B 100%);
    color: #E8ECF3;
}

/* Hide default Streamlit chrome */
#MainMenu, footer {visibility: hidden;}

/* Hero header */
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #F3E7CE;
    margin-bottom: 0.2rem;
    letter-spacing: -0.01em;
}
.hero-sub {
    color: #8B95A7;
    font-size: 1.02rem;
    margin-bottom: 2rem;
}
.eyebrow {
    color: #D4A24E;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}

/* Input card */
.input-card {
    background: #131C2E;
    border: 1px solid #232E45;
    border-radius: 14px;
    padding: 28px 28px 8px 28px;
    margin-bottom: 24px;
}

/* Labels */
label, .stSelectbox label, .stNumberInput label {
    color: #C7CEDB !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* Inputs */
.stSelectbox > div > div, .stNumberInput input {
    background-color: #0F1729 !important;
    border: 1px solid #2A3752 !important;
    border-radius: 8px !important;
    color: #E8ECF3 !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #D4A24E 0%, #B8842E 100%);
    color: #0B1220;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(212, 162, 78, 0.35);
    color: #0B1220;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #17233A 0%, #131C2E 100%);
    border: 1px solid #2A3752;
    border-left: 5px solid #D4A24E;
    border-radius: 14px;
    padding: 26px 30px;
    margin-top: 20px;
}
.result-cluster {
    color: #8B95A7;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.result-name {
    font-family: 'Fraunces', serif;
    font-size: 1.9rem;
    color: #F3E7CE;
    font-weight: 600;
    margin-bottom: 4px;
}
.result-desc {
    color: #A8B1C2;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Footer credit */
.dev-footer {
    text-align: center;
    color: #4A5670;
    font-size: 0.82rem;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #1D2740;
}
.dev-footer span {
    color: #D4A24E;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL + SCALER
# ============================================================
model = joblib.load("mall_kmeans_model.pkl")
scaler = joblib.load("mall_scaler.pkl")

# Cluster metadata: name + short description (edit to match your own analysis)
cluster_info = {
    0: {
        "name": "Cautious Seniors",
        "desc": "Older customers with modest income and conservative spending habits. Best reached with value-driven, trust-based offers."
    },
    1: {
        "name": "High-Income Savers",
        "desc": "Strong earning power but low spending — price alone won't move them. Premium loyalty perks may work better than discounts."
    },
    2: {
        "name": "Young High Spenders",
        "desc": "Younger customers with high spending scores. Prime targets for trend-driven marketing and new arrivals."
    },
    3: {
        "name": "Young Moderate Spenders",
        "desc": "Younger, budget-conscious shoppers with steady but moderate spending. Respond well to seasonal deals and bundles."
    }
}

# ============================================================
# HERO
# ============================================================
st.markdown('<div class="eyebrow">Retail Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Mall Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter a customer\'s profile to see which behavioural segment they belong to — powered by an unsupervised KMeans model.</div>',
    unsafe_allow_html=True
)

# ============================================================
# INPUT FORM
# ============================================================
st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
with col2:
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
    spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

st.markdown('</div>', unsafe_allow_html=True)

predict = st.button("Predict Segment")

# ============================================================
# PREDICTION
# ============================================================
if predict:
    gender_num = 1 if gender == "Male" else 0
    input_data = np.array([[gender_num, age, income, spending]])
    input_scaled = scaler.transform(input_data)

    cluster = int(model.predict(input_scaled)[0])
    info = cluster_info.get(cluster, {"name": f"Cluster {cluster}", "desc": ""})

    st.markdown(f"""
    <div class="result-card">
        <div class="result-cluster">Segment {cluster}</div>
        <div class="result-name">{info['name']}</div>
        <div class="result-desc">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="dev-footer">Built by <span>Aurang Zeb</span> — Machine Learning Project</div>',
    unsafe_allow_html=True
)
