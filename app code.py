import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="ML Classification System", layout="centered")

# --------------------------------------------------
# Simple Login Credentials (academic purpose)
# --------------------------------------------------
USERNAME = "admin"
PASSWORD = "admin123"

# --------------------------------------------------
# Session State for Login
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --------------------------------------------------
# Login Page
# --------------------------------------------------
if not st.session_state.logged_in:

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <h1 style='text-align: center;'>🔐 ML Prediction System</h1>
        <h3 style='text-align: center; color: gray;'>
        Flower & Tumor Classification Platform
        </h3>
        <p style='text-align: center; color: #9e9e9e;'>
        Authorized access required for predictive analysis
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.success("Access granted")
                st.rerun()
            else:
                st.error("Access denied: Invalid credentials")


# --------------------------------------------------
# Main App (After Login)
# --------------------------------------------------
else:

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("Flower and Tumor Classification Project")

    # Load models
    iris_model = joblib.load("iris_model.joblib")
    cancer_bundle = joblib.load("cancer_model.joblib")

    cancer_model = cancer_bundle["model"]
    scaler = cancer_bundle["scaler"]

    # Sidebar Menu
    menu = st.sidebar.selectbox(
        "Select Option",
        [
            "Flower Prediction",
            "Tumor Prediction",
            "Batch Prediction",
            "Prediction History",
            "Model Info"
        ]
    )

    # ==================================================
    # FLOWER PREDICTION
    # ==================================================
    if menu == "Flower Prediction":

        st.subheader("Iris Flower Classification")

        sl = st.number_input("Sepal Length", 5.4)
        sw = st.number_input("Sepal Width", 2.6)
        pl = st.number_input("Petal Length", 4.1)
        pw = st.number_input("Petal Width", 1.3)

        if st.button("Predict Flower"):
            data = [[sl, sw, pl, pw]]
            prediction = iris_model.predict(data)[0]
            prob = iris_model.predict_proba(data)[0]
            confidence = max(prob) * 100

            st.success(f"Predicted Species: {prediction}")
            st.write(f"Confidence: {confidence:.2f}%")

    # ==================================================
    # TUMOR PREDICTION
    # ==================================================
    elif menu == "Tumor Prediction":

        st.subheader("Breast Cancer Prediction")

        r = st.number_input("Radius Mean", 13.45)
        p = st.number_input("Perimeter Mean", 86.6)
        a = st.number_input("Area Mean", 555.1)
        s = st.number_input("Smoothness Mean", 0.1022)
        c = st.number_input("Compactness Mean", 0.08165)
        cn = st.number_input("Concavity Mean", 0.03974)
        sy = st.number_input("Symmetry Mean", 0.1638)

        if st.button("Predict Tumor"):

            row = [[r, p, a, s, c, cn, sy]]
            row_scaled = scaler.transform(row)

            pred = cancer_model.predict(row_scaled)[0]
            prob = cancer_model.predict_proba(row_scaled)[0]
            confidence = max(prob) * 100

            if pred == "M":
                st.error("Result: Malignant")
            else:
                st.success("Result: Benign")

            st.write(f"Confidence: {confidence:.2f}%")

            # Save history
            history_file = "prediction_history.csv"

            record = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "radius_mean": r,
                "perimeter_mean": p,
                "area_mean": a,
                "smoothness_mean": s,
                "compactness_mean": c,
                "concavity_mean": cn,
                "symmetry_mean": sy,
                "prediction": "Malignant" if pred == "M" else "Benign",
                "confidence": round(confidence, 2)
            }])

            if os.path.exists(history_file):
                record.to_csv(history_file, mode="a", header=False, index=False)
            else:
                record.to_csv(history_file, index=False)

    # ==================================================
    # BATCH PREDICTION
    # ==================================================
    elif menu == "Batch Prediction":

        st.subheader("Batch Tumor Prediction (CSV)")

        file = st.file_uploader("Upload CSV file", type=["csv"])

        if file:
            df = pd.read_csv(file)
            st.write("Input Data", df.head())

            scaled = scaler.transform(df)
            preds = cancer_model.predict(scaled)
            probs = cancer_model.predict_proba(scaled)

            df["Prediction"] = np.where(preds == "M", "Malignant", "Benign")
            df["Confidence (%)"] = np.max(probs, axis=1) * 100

            st.write("Results", df)

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "batch_predictions.csv",
                "text/csv"
            )

    # ==================================================
    # HISTORY
    # ==================================================
    elif menu == "Prediction History":

        st.subheader("Prediction History")

        if os.path.exists("prediction_history.csv"):
            history_df = pd.read_csv("prediction_history.csv")
            st.dataframe(history_df)
        else:
            st.info("No history available")

    # ==================================================
    # MODEL INFO
    # ==================================================
    else:

        st.subheader("Model Information")

        st.write("Algorithm: Logistic Regression")
        st.write("Datasets: Iris, Breast Cancer")
        st.write("Accuracy:")
        st.write("- Flower: ~97%")
        st.write("- Cancer: ~96%")

        st.info(
            "This application is for academic and learning purposes only "
            "and not intended for real medical diagnosis."
        )

