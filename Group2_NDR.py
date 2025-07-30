import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Loan Amount Predictor - Group 2", layout="wide")

# ------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ------------------------------------------------
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Predictor", "About"],
        icons=["house", "calculator", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#fff6e5"},
            "icon": {"color": "#d97706", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "--hover-color": "#fde68a"},
            "nav-link-selected": {"background-color": "#fcd34d", "color": "#000"},
        }
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Loan Amount Predictor - Group 2")
    st.write("This tool predicts the expected loan amount a customer may receive based on their details.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("📊 Predict Loan Amount")
    st.write("Enter the customer's details below to estimate their eligible loan amount.")

    input_data = {}

    # Collect inputs with proper widgets
    for col in X_columns:
        if col == 'age':
            input_data[col] = st.number_input("Age", min_value=0, value=30)
        elif col == 'income':
            input_data[col] = st.number_input("Income (GHS)", min_value=0.0, value=1000.0)
        elif col == 'credit_score':
            input_data[col] = st.number_input("Credit Score", min_value=0.0, value=600.0)
        elif col == 'loan_term':
            input_data[col] = st.number_input("Loan Term (Months)", min_value=1.0, value=12.0)
        elif col == 'gender':
            gender = st.selectbox("Gender", ['Male', 'Female'])
            input_data[col] = 1 if gender == 'Male' else 0
        elif col == 'employment_type':
            emp_type = st.selectbox("Employment Type", ['Salaried', 'Self-employed', 'Unemployed'])
            emp_map = {'Salaried': 0, 'Self-employed': 1, 'Unemployed': 2}
            input_data[col] = emp_map[emp_type]

    # Convert to DataFrame and scale
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predict
    if st.button("Predict Loan Amount"):
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Predicted Loan Amount: GHS {prediction:,.2f}")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
        This app was developed by Group 2 to predict the loan amount a customer might receive,
        using a Decision Tree Regressor trained on key customer data.

        Tools Used: Python, Streamlit, Scikit-learn  
        Target Variable: Loan Amount  
        Purpose: Estimate realistic loan allocations for Xente customers.
    """)
