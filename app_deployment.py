import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="centered"
)

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load('employee_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

model, feature_names = load_model()

# --- App header and introduction ---
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>👥 Employee Attrition Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
st.write("Use this smart tool to instantly predict employee attrition and risk levels with just their basic details.")

# --- Inputs in grid ---
st.markdown("### Enter Employee Details")
col1a, col1b = st.columns(2)
with col1a:
    age = st.slider("Age", 20, 40, 28)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
with col1b:
    city = st.selectbox("City", ["Bangalore", "Pune", "New Delhi"], index=0)
    education = st.selectbox("Education Level", ["Bachelors", "Masters", "PHD"], index=0)

col2a, col2b = st.columns(2)
with col2a:
    joining_year = st.slider("Joining Year", 2005, 2023, 2020)
    payment_tier = st.selectbox("Payment Tier (1=Low, 3=High)", [1, 2, 3], index=0)
with col2b:
    experience = st.slider("Domain Experience (years)", 0, 10, 2)
    ever_benched = st.selectbox("Ever Benched?", ["Yes", "No"], index=1)

# --- Predict and show results ---
if st.button("🔮 Predict Attrition Risk"):
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    input_df['Age'] = age
    input_df['Gender'] = 1 if gender == "Male" else 0
    input_df['ExperienceInCurrentDomain'] = experience
    input_df['PaymentTier'] = payment_tier
    input_df['YearsInCompany'] = datetime.now().year - joining_year
    input_df['EverBenched'] = 1 if ever_benched == "Yes" else 0

    # One-hot columns
    education_col = f"Education_{education}"
    city_col = f"City_{city}"
    age_group = '20-25' if age <= 25 else '26-30' if age <= 30 else '31-35' if age <= 35 else '36-40'
    age_group_col = f"AgeGroup_{age_group}"
    exp_level = 'Fresher(0-1)' if experience <= 1 else 'Mid(2-3)' if experience <= 3 else 'Senior(4+)'
    exp_level_col = f"ExpLevel_{exp_level}"

    for col in [education_col, city_col, age_group_col, exp_level_col]:
        if col in input_df.columns:
            input_df[col] = 1

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Result block in two columns
    st.markdown("---")
    st.markdown("## Prediction Result")
    rescol1, rescol2 = st.columns(2)

    with rescol1:
        if prediction == 1:
            st.error("**⚠️ Likely to LEAVE**")
            st.metric(label="Attrition Probability", value=f"{probability:.2f}", delta="HIGH")
        else:
            st.success("**✅ Likely to STAY**")
            st.metric(label="Attrition Probability", value=f"{probability:.2f}", delta="LOW")
        st.write("")

    with rescol2:
        st.write("> **Probability Gauge:**")
        st.progress(probability)
        st.write(f"0 = Minimum risk | 1 = Maximum risk\n\n")

    # Details in expandable box for compact look
    with st.expander("Show Prediction Details"):
        st.write(f"**Employee Info:** Age: {age}, Gender: {gender}, City: {city}, Education: {education}")
        st.write(f"**Professional Info:** Payment Tier: {payment_tier}, Experience: {experience}, Years in Company: {datetime.now().year - joining_year}, Ever Benched: {ever_benched}")
        st.write(f"**Prediction:** {'Leave' if prediction == 1 else 'Stay'}")
        st.write(f"**Attrition Probability:** {probability:.2f}")

    # Download report
    result_df = pd.DataFrame({
        "Feature": ["Age", "Gender", "City", "Education", "Payment Tier", "Experience", "Ever Benched", "Prediction", "Probability"],
        "Value": [age, gender, city, education, payment_tier, experience, ever_benched, "Leave" if prediction == 1 else "Stay", round(probability, 2)]
    })
    st.download_button(
        label="📥 Download Prediction Report",
        data=result_df.to_csv(index=False),
        file_name="employee_attrition_prediction.csv",
        mime="text/csv"
    )

    # Informative tip for user
    if probability > 0.7:
        st.warning("This employee has a high probability of attrition. You may want to consider retention strategies.")
    elif probability > 0.4:
        st.info("Moderate risk. Regular engagement and feedback may help retain this employee.")
    else:
        st.success("Low attrition risk detected. Keep up your employee engagement practices!")

else:
    st.info("Fill all employee details and click **Predict Attrition Risk** for results.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit • Employee Attrition Analysis © 2025")




