# NHANES-Based Obesity Risk Score

## Overview

This project develops and validates a composite Obesity Risk Score using select anthropometric, clinical, behavioral, and socioeconomic variables from the NHANES dataset. The scoring model applies evidence-based weights to normalized feature inputs, generating an interpretable risk score for each participant.

A live demo of the risk calculator is available here:  
https://obesity-risk-calculator.streamlit.app/

## Key Features

- **Factor Selection:** Incorporates BMI, waist circumference, A1C, blood pressure, triglycerides, total cholesterol, alcohol use, food security, weight history, smoking, physical activity, and HDL.
- **Score Construction:** All features are normalized to a 0–1 scale and combined via a weighted sum reflecting clinical relevance and risk direction (protective vs. risk).
- **Interpretability:** The score is presented on a 0–100 scale and categorized as Low, Moderate, or High risk using clearly defined cutpoints.
- **Visualization:** Includes score distribution plots and basic risk category summaries.
- **AI Explanation (Streamlit app):** Integration with OpenAI's GPT-4 model to generate personalized plain-language explanations and actionable recommendations.

## Methodology (High Level)

- Extract relevant variables from NHANES (2017–2020) including anthropometrics, labs, questionnaires, and behavioral measures.
- Clean and recode variables (e.g., hypertension, smoking, food security, weight history) into consistent risk-oriented formats.
- Normalize each feature to a 0–1 scale and apply a weighted linear combination:
  - Higher weights on BMI, waist, A1C, and hypertension.
  - Moderate weights on triglycerides and total cholesterol.
  - Lower weights on alcohol use, food security, weight history, and smoking.
  - Negative weights on HDL and physical activity to reflect protective effects.
- Convert the composite score to a 0–100 scale and map to Low / Moderate / High risk categories.

## Quick Insights

- The choice of normalization (sample-based vs clinical cutpoints) has a large effect on the score distribution and proportion of users flagged as high risk.
- BMI, waist circumference, and A1C contribute most strongly to score variation across individuals.
- Protective factors such as HDL and physical activity moderately reduce risk but rarely fully offset high BMI or poor metabolic markers.
- Socioeconomic and behavioral measures (food security, alcohol, smoking, recent weight gain) provide additional nuance on top of core clinical factors.

## Run Locally 
- install requirements.txt, and run streamlit app with - streamlit run streamlit_app.py

1. Install dependencies:

