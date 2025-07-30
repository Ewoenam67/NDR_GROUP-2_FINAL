import streamlit as st 
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Disaster Impact Predictor - Group 2", layout="wide")

# ------------------------------------------------
# LOAD TRAINED ARTIFACTS
# ------------------------------------------------
model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------------------------
# SIDEBAR NAVIGATION (Light mode fix with custom title)
# ------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h3 style='margin-bottom: 10px; color: #92400e;'>📑 Navigation</h3>",
        unsafe_allow_html=True
    )
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#fef3c7"},
            "icon": {"color": "#92400e", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "color": "#000000",
                "--hover-color": "#fde68a"
            },
            "nav-link-selected": {
                "background-color": "#fcd34d",
                "color": "#000000"
            },
        }
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Disaster Impact Predictor - Group 2")
    st.write("This tool predicts the number of people potentially affected by a natural disaster using historical patterns.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("📊 Disaster Impact Prediction")
    st.write("Provide the disaster and location details to estimate the number of people affected.")

    input_data = {}

    # Example mappings (update as per your preprocessing!)
    disaster_group_options = {"Biological": 0, "Climatological": 1, "Geophysical": 2, "Hydrological": 3, "Meteorological": 4, "Technological": 5}
    disaster_type_options = {"Flood": 0, "Earthquake": 1, "Storm": 2, "Epidemic": 3}
    country_options = {"Ghana": 0, "Nigeria": 1, "Kenya": 2, "South Africa": 3}
    region_options = {"Middle Africa": 0, "Eastern Africa": 1, "Western Africa": 2, "Cental Africa": 3, "Southern Africa": 4}

    if 'Year' in X_columns:
        input_data['Year'] = st.number_input("Year", min_value=1900, max_value=2100, value=2023)

    for col in X_columns:
        if col == 'Year':
            continue
        elif col.lower() == 'disaster_group':
            selection = st.selectbox("Disaster Group", list(disaster_group_options.keys()))
            input_data[col] = disaster_group_options[selection]
        elif col.lower() == 'disaster_type':
            selection = st.selectbox("Disaster Type", list(disaster_type_options.keys()))
            input_data[col] = disaster_type_options[selection]
        elif col.lower() == 'country':
            selection = st.selectbox("Country", list(country_options.keys()))
            input_data[col] = country_options[selection]
        elif col.lower() == 'region':
            selection = st.selectbox("Region", list(region_options.keys()))
            input_data[col] = region_options[selection]
        elif col.lower() in ['total_deaths', 'number_injured', 'number_affected', 'number_homeless']:
            input_data[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, value=100.0)
        else:
            input_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", value=0.0)

    # Convert to DataFrame and align with training structure
    input_df = pd.DataFrame([input_data])

    for col in X_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_columns]

    try:
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        if st.button("Predict Affected People"):
            prediction = model.predict(input_scaled)[0]
            st.success(f"📌 Estimated Number of People Affected: {prediction:,.0f}")
    except Exception as e:
        st.error("🚫 Prediction Failed.")
        st.code(str(e))
        st.write("Expected columns:", X_columns)
        st.write("Current input shape:", input_df.shape)

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
        This app was developed by Group 2 to estimate the number of people affected by natural disasters.

        *Model Used*: Decision Tree Regressor  
        *Tools*: Python, Scikit-learn, Streamlit  
        *Dataset*: Natural Disaster Records (1993–2023)  
        *Goal*: Provide insights for emergency preparedness and resource planning.
    """)
