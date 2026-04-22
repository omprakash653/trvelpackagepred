import streamlit as st
import pandas as pd
import joblib

import mysql.connector

# DB Connection ######################################
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="shivam35",
        database="ml_app"
    )

def save_to_db(data, prediction, probability):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO predictions (
        Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender,
        NumberOfFollowups, ProductPitched, PreferredPropertyStar, MaritalStatus,
        NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar, Designation,
        MonthlyIncome, TotalVisiting, Prediction, Probability
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
    int(data["Age"][0]),
    str(data["TypeofContact"][0]),
    int(data["CityTier"][0]),
    int(data["DurationOfPitch"][0]),
    str(data["Occupation"][0]),
    str(data["Gender"][0]),
    int(data["NumberOfFollowups"][0]),
    str(data["ProductPitched"][0]),
    int(data["PreferredPropertyStar"][0]),
    str(data["MaritalStatus"][0]),
    int(data["NumberOfTrips"][0]),
    int(data["Passport"][0]),
    int(data["PitchSatisfactionScore"][0]),
    int(data["OwnCar"][0]),
    str(data["Designation"][0]),
    int(data["MonthlyIncome"][0]),
    int(data["TotalVisiting"][0]),
    int(prediction),                         # ✅ FIX
    float(probability) if probability else None  # ✅ FIX
)

    cursor.execute(query, values)
    conn.commit()
    conn.close()

############################################################

# Load model and preprocessor
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# Page config
st.set_page_config(page_title="Product Taken Prediction", layout="centered")

st.title("🧳 Product Taken Prediction App")
st.write("Enter customer details to predict whether the product will be taken or not.")

st.markdown("---")

# =========================
# Input fields
# =========================

Age = st.number_input("Age", min_value=18, max_value=100, value=30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox("City Tier", [1, 2, 3])

DurationOfPitch = st.number_input(
    "Duration Of Pitch (minutes)", min_value=0, max_value=300, value=30
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

NumberOfFollowups = st.number_input(
    "Number Of Followups", min_value=0, max_value=20, value=2
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Unmarried", "Married", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number Of Trips", min_value=0, max_value=50, value=2
)

Passport = st.selectbox("Passport", [0, 1])

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score", min_value=1, max_value=5, value=3
)

OwnCar = st.selectbox("Own Car", [0, 1])

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income", min_value=0, value=50000
)

TotalVisiting = st.number_input(
    "Total Visiting", min_value=0, max_value=50, value=2
)

# =========================
# Prediction
# =========================

if st.button("🔍 Predict"):
    input_data = pd.DataFrame({
        "Age": [Age],
        "TypeofContact": [TypeofContact],
        "CityTier": [CityTier],
        "DurationOfPitch": [DurationOfPitch],
        "Occupation": [Occupation],
        "Gender": [Gender],
        "NumberOfFollowups": [NumberOfFollowups],
        "ProductPitched": [ProductPitched],
        "PreferredPropertyStar": [PreferredPropertyStar],
        "MaritalStatus": [MaritalStatus],
        "NumberOfTrips": [NumberOfTrips],
        "Passport": [Passport],
        "PitchSatisfactionScore": [PitchSatisfactionScore],
        "OwnCar": [OwnCar],
        "Designation": [Designation],
        "MonthlyIncome": [MonthlyIncome],
        "TotalVisiting": [TotalVisiting]
    })

    X_processed = preprocessor.transform(input_data) 
    prediction = model.predict(X_processed)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_processed)[0][1]
        st.info(f"📊 Probability: {prob:.2f}")

    if prediction == 1:
        st.success("✅ Product WILL BE TAKEN")
    else:
        st.error("❌ Product WILL NOT BE TAKEN")

    # ✅ SAVE TO DATABASE
    save_to_db(input_data, prediction, prob)

def fetch_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

st.markdown("## 📂 Stored Predictions")

if st.button("📊 View Saved Data"):
    df = fetch_data()
    st.dataframe(df)

st.markdown("---")
st.caption("ML Model Prediction using Streamlit")
