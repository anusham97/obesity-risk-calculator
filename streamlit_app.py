import streamlit as st
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

NHANES_STATS = {
    "BMI": {"min": 15.0, "max": 50.0},
    "Waist": {"min": 25.0, "max": 70.0},  # inches
    "A1C": {"min": 4.0, "max": 13.0},
    "Trigly": {"min": 30.0, "max": 600.0},
    "TotChol": {"min": 90.0, "max": 350.0},
    "Hypertension": {"min": 0.0, "max": 1.0},
    "ALC": {"min": 0.0, "max": 21.0},
    "FoodSec": {"min": 0.0, "max": 3.0},
    "WeightHist": {"min": 0.0, "max": 1.0},
    "Smoking": {"min": 0.0, "max": 1.0},
    "PhysAct": {"min": 0.0, "max": 2000.0},
    "HDL": {"min": 20.0, "max": 100.0}
}

WEIGHTS = {
    "BMI_norm": 0.225, "Waist_norm": 0.225, "A1C_norm": 0.175, "Trigly_norm": 0.10, "TotChol_norm": 0.10,
    "Hypertension_norm": 0.175, "ALC_norm": 0.05, "FoodSec_norm": 0.05, "WeightHist_norm": 0.05,
    "Smoking_norm": 0.05, "PhysAct_norm": -0.10, "HDL_norm": -0.10
}

def min_max_norm(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def compute_obesity_risk(user_inputs):
    norm = {
        "BMI_norm": min_max_norm(user_inputs["BMI"], *NHANES_STATS["BMI"].values()),
        "Waist_norm": min_max_norm(user_inputs["Waist"], *NHANES_STATS["Waist"].values()),
        "A1C_norm": min_max_norm(user_inputs["A1C"], *NHANES_STATS["A1C"].values()),
        "Trigly_norm": min_max_norm(user_inputs["Trigly"], *NHANES_STATS["Trigly"].values()),
        "TotChol_norm": min_max_norm(user_inputs["TotChol"], *NHANES_STATS["TotChol"].values()),
        "Hypertension_norm": min_max_norm(user_inputs["Hypertension"], *NHANES_STATS["Hypertension"].values()),
        "ALC_norm": min_max_norm(user_inputs["ALC"], *NHANES_STATS["ALC"].values()),
        "FoodSec_norm": min_max_norm(user_inputs["FoodSec"], *NHANES_STATS["FoodSec"].values()),
        "WeightHist_norm": min_max_norm(user_inputs["WeightHist"], *NHANES_STATS["WeightHist"].values()),
        "Smoking_norm": min_max_norm(user_inputs["Smoking"], *NHANES_STATS["Smoking"].values()),
        "PhysAct_norm": min_max_norm(user_inputs["PhysAct"], *NHANES_STATS["PhysAct"].values()),
        "HDL_norm": min_max_norm(user_inputs["HDL"], *NHANES_STATS["HDL"].values()),
    }
    raw_score = sum(norm[k] * WEIGHTS[k] for k in WEIGHTS)
    score_100 = float(np.clip(raw_score, 0, 1) * 100)
    return score_100, norm

def interpret_risk_zone(score):
    if score < 20:
        return "Low", "🟢 Low Risk — Maintain your current habits!"
    elif score < 50:
        return "Moderate", "🟠 Moderate Risk — You may benefit from lifestyle adjustments."
    else:
        return "High", "🔴 High Risk — Consider clinical follow-up + lifestyle changes."

def ai_score_reasoning(score, risk_cat, user_norm_inputs):
    factor_msgs = [
        f"{k}: {v:.2f} (weight {WEIGHTS[k]:+.2f})"
        for k, v in user_norm_inputs.items()
    ]
    factors_str = "; ".join(factor_msgs)
    prompt = (
        f"A user receives an obesity risk score of {score:.1f}/100 (category: {risk_cat}). "
        f"Their normalized features are: {factors_str}. "
        "Write a brief, plain-English explanation for this score, focusing on the biggest drivers and 2–3 actionable steps. "
        "After giving the explanation, give suggestions to lower risk or continue good track. (Max 180 words.)"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a health coach explaining metabolic risk."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=350
    )
    return response.choices[0].message.content

st.title("🩺 Obesity Risk Score Calculator")

with st.form("obesity_form"):
    col1, col2 = st.columns(2)
    with col1:
        BMI = st.number_input(
            "BMI (kg/m²)", value=29.5, step=0.1,
            help="Example: 22 = normal, 27 = overweight, 32 = obese"
        )
        Waist = st.number_input(
            "Waist Circumference (inches)", value=44.0, step=0.1,
            help="Example: 32\" (slim) to 50\" (high)"
        )
        A1C = st.number_input(
            "A1C (%)", value=5.8, step=0.1,
            help="Three-month blood sugar average"
        )
        Trigly = st.number_input(
            "Triglycerides (mg/dL)", value=150.0, step=5.0,
            help="<150 is normal"
        )
        TotChol = st.number_input(
            "Total Cholesterol (mg/dL)", value=185.0, step=5.0,
            help="Desirable <200"
        )
        Hypertension = st.selectbox(
            "Ever told you have high blood pressure? (BPQ030)", 
            [0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="NHANES self-report"
        )

    with col2:
        ALC = st.number_input(
            "Alcohol — drinks per week", value=3.0, step=0.5,
            help="General weekly alcohol estimate"
        )
        FoodSec = st.select_slider(
            "Food Security (FSDHH)", options=[0, 1, 2, 3], value=1,
            format_func=lambda x: ["Full", "Marginal", "Low", "Very Low"][x] + " Security",
            help="0=Full, 3=Very Low"
        )
        WeightHist = st.selectbox(
            "Gained 10+ lbs in past year?", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        Smoking = st.selectbox(
            "Currently smoke?", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        PhysAct = st.number_input(
            "Weekly Physical Activity (minutes)", value=120.0, step=10.0,
            help="Typical vigorous or moderate activity minutes"
        )
        HDL = st.number_input(
            "HDL Cholesterol (mg/dL)", value=48.0, step=1.0,
            help="Higher is generally better"
        )

    submitted = st.form_submit_button("Calculate My Score")

if submitted:
    user_inputs = {
        "BMI": BMI,
        "Waist": Waist,
        "A1C": A1C,
        "Trigly": Trigly,
        "TotChol": TotChol,
        "Hypertension": Hypertension,
        "ALC": ALC,
        "FoodSec": FoodSec,
        "WeightHist": WeightHist,
        "Smoking": Smoking,
        "PhysAct": PhysAct,
        "HDL": HDL,
    }

    score, norm_inputs = compute_obesity_risk(user_inputs)
    risk_cat, risk_msg = interpret_risk_zone(score)

    st.subheader(f"Your Score: **{score:.1f}/100** — {risk_cat} Risk")
    st.write(risk_msg)

    st.write("### Personalized AI Interpretation")
    with st.spinner("Querying GPT..."):
        explanation = ai_score_reasoning(score, risk_cat, norm_inputs)
        st.write(explanation)
