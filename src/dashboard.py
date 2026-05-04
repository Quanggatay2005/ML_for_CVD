"""
src/dashboard.py -- Streamlit Frontend for CVD Risk Predictor
=============================================================
Connects to the FastAPI backend at http://localhost:8000.
Provides a rich, user-friendly UI for entering patient data
and visualising the predicted CVD risk.
"""

import streamlit as st
import requests
import json
import os
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Human-readable descriptions for every feature in new_brfss.csv ────────────
# Format: feature_name -> (label, widget_type, *widget_args)
#   selectbox  : (label, "selectbox", [values], [display_labels])
#   slider     : (label, "slider",    min, max, default)
#   number     : (label, "number",    min, max, default, step)

FEATURE_UI = {
    # Demographics
    "_AGEG5YR": (
        "Age Group", "selectbox",
        [1,2,3,4,5,6,7,8,9,10,11,12,13],
        ["18-24","25-29","30-34","35-39","40-44","45-49",
         "50-54","55-59","60-64","65-69","70-74","75-79","80+"],
    ),
    "SEXVAR": (
        "Sex", "selectbox",
        [1, 2],
        ["Male", "Female"],
    ),
    "_IMPRACE": (
        "Race / Ethnicity", "selectbox",
        [1,2,3,4,5,6],
        ["White non-Hispanic","Black non-Hispanic","Asian non-Hispanic",
         "American Indian / Alaska Native","Hispanic","Other race"],
    ),
    "_EDUCAG": (
        "Education Level", "selectbox",
        [1,2,3,4],
        ["Did not graduate high school","Graduated high school",
         "Attended college / tech school","Graduated college / tech school"],
    ),
    "_INCOMG1": (
        "Household Income", "selectbox",
        [1,2,3,4,5,6,7,9],
        ["< $15,000","$15,000-$25,000","$25,000-$35,000","$35,000-$50,000",
         "$50,000-$100,000","$100,000-$200,000","$200,000+","Don't know / Refused"],
    ),
    # General Health
    "_HLTHPLN": (
        "Have Health Insurance?", "selectbox",
        [1, 2],
        ["Yes", "No"],
    ),
    "GENHLTH": (
        "General Health Rating", "selectbox",
        [1,2,3,4,5],
        ["Excellent","Very Good","Good","Fair","Poor"],
    ),
    "PHYSHLTH": (
        "Poor Physical Health Days (past 30 days)", "slider",
        0, 30, 0,
    ),
    "MENTHLTH": (
        "Poor Mental Health Days (past 30 days)", "slider",
        0, 30, 0,
    ),
    # CVD Risk
    "CVDCRHD4": (
        "Ever diagnosed with coronary heart disease?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    "CVDSTRK3": (
        "Ever had a stroke?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    # Diabetes & BMI
    "DIABETE4": (
        "Diabetes Status", "selectbox",
        [3, 2, 1, 4],
        ["No","Prediabetes","Yes (diabetes)","Gestational diabetes"],
    ),
    "_BMI5CAT": (
        "BMI Category", "selectbox",
        [1, 2, 3, 4],
        ["Underweight (< 18.5)","Normal weight (18.5-24.9)",
         "Overweight (25-29.9)","Obese (>= 30)"],
    ),
    "_BMI5": (
        "BMI (actual value x 100, e.g. 2500 = BMI 25.0)", "number",
        1000, 9999, 2500, 1,
    ),
    # Lifestyle
    "_TOTINDA": (
        "Physical Activity in Past 30 Days?", "selectbox",
        [1, 2],
        ["Yes", "No"],
    ),
    "SLEPTIM1": (
        "Average Hours of Sleep per Night", "slider",
        1, 24, 7,
    ),
    "SMOKE100": (
        "Smoked at Least 100 Cigarettes in Lifetime?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    "_RFSMOK3": (
        "Current Smoker?", "selectbox",
        [1, 2],
        ["No", "Yes"],
    ),
    "_RFDRHV8": (
        "Heavy Alcohol Drinker?", "selectbox",
        [1, 2],
        ["No", "Yes"],
    ),
    # Comorbidities
    "CHCCOPD3": (
        "Ever diagnosed with COPD or Emphysema?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    "ADDEPEV3": (
        "Ever diagnosed with Depressive Disorder?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    "CHCKDNY2": (
        "Ever diagnosed with Kidney Disease?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    "HAVARTH4": (
        "Told have Arthritis?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    "CHCOCNC1": (
        "Ever diagnosed with non-skin Cancer?", "selectbox",
        [2, 1],
        ["No", "Yes"],
    ),
    # Preventive Care
    "CHECKUP1": (
        "Last Routine Checkup", "selectbox",
        [1,2,3,4,8],
        ["Within past year","Within past 2 years","Within past 5 years",
         "5 or more years ago","Never"],
    ),
    "EXERANY2": (
        "Any Exercise in Past 30 Days?", "selectbox",
        [1, 2],
        ["Yes", "No"],
    ),
    # Derived flags (0/1)
    "POOR_HEALTH": (
        "Poor/Very Poor General Health?", "selectbox",
        [0, 1],
        ["No", "Yes"],
    ),
    "OBESE": (
        "Obese (BMI >= 30)?", "selectbox",
        [0, 1],
        ["No", "Yes"],
    ),
    "SMOKER": (
        "Current Smoker (flag)?", "selectbox",
        [0, 1],
        ["No", "Yes"],
    ),
    "INACTIVE": (
        "Physically Inactive?", "selectbox",
        [0, 1],
        ["No", "Yes"],
    ),
    "SHORT_SLEEP": (
        "Short Sleeper (< 6 hrs)?", "selectbox",
        [0, 1],
        ["No", "Yes"],
    ),
    "DIABETES": (
        "Diabetes (flag)?", "selectbox",
        [0, 1],
        ["No", "Yes"],
    ),
    "HYPERTENSION": (
        "Hypertension (High Blood Pressure)?", "selectbox",
        [0, 1],
        ["No", "Yes"],
    ),
}


def render_widget(feature, col):
    """Render a Streamlit input widget for the given feature inside `col`."""
    with col:
        if feature in FEATURE_UI:
            info = FEATURE_UI[feature]
            label, wtype = info[0], info[1]
            rest = info[2:]

            if wtype == "selectbox":
                values, labels = rest[0], rest[1]
                chosen_label = st.selectbox(label, options=labels, key=feature)
                return values[labels.index(chosen_label)]

            elif wtype == "slider":
                min_v, max_v, default_v = rest
                return st.slider(label, min_value=min_v, max_value=max_v,
                                 value=default_v, key=feature)

            elif wtype == "number":
                min_v, max_v, default_v, step = rest
                return st.number_input(label, min_value=float(min_v),
                                       max_value=float(max_v),
                                       value=float(default_v),
                                       step=float(step), key=feature)
        # Fallback for any unlisted feature
        return st.number_input(feature, value=0.0, step=1.0, key=feature)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CVD Risk Predictor")
    st.divider()

    # API health
    st.markdown("### Backend Status")
    try:
        health = requests.get(f"{API_URL}/", timeout=3).json()
        if health.get("model_loaded"):
            st.success(f"API Online")
        else:
            st.warning("API online but model not loaded.\nRun `python main.py --no-api` first.")
    except Exception:
        st.error("API Offline. Start with:\n```\npython main.py --api-only\n```")


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("Cardiovascular Disease Risk Predictor")
st.markdown(
    "Enter the patient's health indicators below and click **Predict CVD Risk** "
    "to estimate the probability of heart attack history."
)

# Fetch feature list from API
try:
    feat_resp = requests.get(f"{API_URL}/features", timeout=3).json()
    feature_names = feat_resp.get("features", [])
except Exception:
    feature_names = []

if not feature_names:
    st.error(
        "Cannot load feature list from API. "
        "Make sure the backend is running:\n```\npython main.py --api-only\n```"
    )
    st.stop()

# ── Render feature inputs ─────────────────────────────────────────────────────
st.subheader("Patient Information")

# Group into sections for clarity
groups = {
    "Demographics": ["_AGEG5YR","SEXVAR","_IMPRACE","_EDUCAG","_INCOMG1"],
    "General Health": ["GENHLTH","PHYSHLTH","MENTHLTH","_HLTHPLN"],
    "Cardiovascular Risk": ["CVDCRHD4","CVDSTRK3"],
    "Diabetes & BMI": ["DIABETE4","_BMI5CAT","_BMI5"],
    "Lifestyle": ["_TOTINDA","SLEPTIM1","SMOKE100","_RFSMOK3","_RFDRHV8"],
    "Comorbidities": ["CHCCOPD3","ADDEPEV3","CHCKDNY2","HAVARTH4","CHCOCNC1"],
    "Preventive Care": ["CHECKUP1","EXERANY2"],
    "Derived Flags": ["POOR_HEALTH","OBESE","SMOKER","INACTIVE",
                      "SHORT_SLEEP","DIABETES","HYPERTENSION"],
}

# Build a flat list of features in grouped order (preserving any extras)
grouped_features = [f for group in groups.values() for f in group]
remaining = [f for f in feature_names if f not in grouped_features]

patient_data = {}

for section, feats in groups.items():
    present = [f for f in feats if f in feature_names]
    if not present:
        continue

    with st.expander(f"**{section}**", expanded=(section in ["Demographics","General Health","Cardiovascular Risk"])):
        cols = st.columns(min(3, len(present)))
        for i, feature in enumerate(present):
            patient_data[feature] = render_widget(feature, cols[i % len(cols)])

# Any extra features not in the groups above
if remaining:
    with st.expander("**Other Features**", expanded=False):
        cols = st.columns(3)
        for i, feature in enumerate(remaining):
            patient_data[feature] = render_widget(feature, cols[i % 3])

# ── Predict button ────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button(
    "Predict CVD Risk",
    type="primary",
    use_container_width=True,
    key="predict_btn",
)

if predict_btn:
    with st.spinner("Analysing patient data..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                risk_pct   = result.get("risk_percentage", 0)
                risk_level = result.get("risk_level", "Unknown")

                st.divider()
                st.subheader("Prediction Result")

                col_risk, col_prob = st.columns(2)
                with col_risk:
                    if risk_level == "High":
                        st.error(f"**Risk Level: {risk_level}**")
                    else:
                        st.success(f"**Risk Level: {risk_level}**")
                with col_prob:
                    st.metric("Estimated Risk", f"{risk_pct:.1f}%")

                # Progress bar
                prog_val = max(0, min(100, int(risk_pct)))
                st.progress(prog_val / 100)

                st.info(result.get("interpretation", ""))

                if risk_level == "High":
                    st.warning(
                        "A HIGH risk prediction suggests the patient may have reported "
                        "a history of heart attack. Clinical evaluation is strongly advised."
                    )

            elif response.status_code == 422:
                detail = response.json().get("detail", response.text)
                st.error(f"Validation error: {detail}")
            else:
                st.error(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the prediction API. "
                "Start it with:  `python main.py --api-only`"
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")
