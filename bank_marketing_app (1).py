import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Page config
st.set_page_config(page_title="Bank Term Deposit Prediction", layout="centered")
st.title("💰 Bank Marketing Term Deposit Prediction")
st.markdown("Enter customer details to predict if they will subscribe to a term deposit.")

# Sidebar inputs (categorical values shown to user)
st.sidebar.header("📋 Customer Information")
age = st.sidebar.slider("Age", 18, 95, 30)
job = st.sidebar.selectbox("Job", label_encoders["job"].classes_)
marital = st.sidebar.selectbox("Marital Status", label_encoders["marital"].classes_)
education = st.sidebar.selectbox("Education", label_encoders["education"].classes_)
default = st.sidebar.selectbox("Has Credit in Default?", label_encoders["default"].classes_)
housing = st.sidebar.selectbox("Has Housing Loan?", label_encoders["housing"].classes_)
loan = st.sidebar.selectbox("Has Personal Loan?", label_encoders["loan"].classes_)
contact = st.sidebar.selectbox("Contact Communication Type", label_encoders["contact"].classes_)
month = st.sidebar.selectbox("Last Contact Month", label_encoders["month"].classes_)
day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)
duration = st.sidebar.slider("Call Duration (in seconds)", 0, 3000, 300)
campaign = st.sidebar.slider("Number of Contacts During Campaign", 1, 50, 1)
pdays = st.sidebar.slider("Days Since Last Contact", -1, 999, -1)
previous = st.sidebar.slider("Previous Contacts", 0, 10, 0)
poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", label_encoders["poutcome"].classes_)
balance = st.sidebar.number_input("Account Balance", value=1000)

# Build readable input dictionary
input_data = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

# ✅ Display readable customer input
if st.checkbox("Show Customer Details (Readable)"):
    st.subheader("🧾 Customer Input Summary")
    st.dataframe(pd.DataFrame([input_data]))

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ✅ Encode only after displaying raw input
for col in input_df.select_dtypes(include="object").columns:
    if col in label_encoders:
        try:
            input_df[col] = label_encoders[col].transform(input_df[col])
        except ValueError as e:
            st.error(f"Encoding error in '{col}': {e}")
            st.stop()
    else:
        st.error(f"Missing encoder for: {col}")
        st.stop()

# Optional: show encoded values
if st.checkbox("Show Encoded Input (For Debugging)"):
    st.subheader("🔢 Encoded Input Data")
    st.dataframe(input_df)

# Match model input columns
try:
    input_df = input_df[model.feature_names_in_]
except Exception as e:
    st.error(f"Column alignment failed: {e}")
    st.stop()

# 🔮 Prediction & Download
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        result_label = "Yes" if pred == 1 else "No"
        confidence = f"{proba:.2%}"

        # Display result
        if pred == 1:
            st.success(f"✅ Customer will likely SUBSCRIBE (Confidence: {confidence})")
        else:
            st.warning(f"❌ Customer will likely NOT subscribe (Confidence: {confidence})")

        # Prepare CSV for download
        result_data = input_data.copy()
        result_data["Prediction"] = result_label
        result_data["Confidence"] = confidence
        result_df = pd.DataFrame([result_data])
        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Result as CSV",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# 📊 Optional: Feature importance
if st.checkbox("Show Feature Importances"):
    importances = model.feature_importances_
    features = model.feature_names_in_
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=features, ax=ax)
    ax.set_title("🔍 Feature Importance")
    st.pyplot(fig)
