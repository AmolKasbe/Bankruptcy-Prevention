import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
# Assume dataset is in a CSV file named 'risk_data.csv'
df = pd.read_csv("bank.csv")
X = df.drop(columns=[" class"])
y = df[" class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVC model
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)

# Save model using pickle
with open("svc.pkl", "wb") as model_file:
    pickle.dump(svc, model_file)

# Load trained model
with open("svc.pkl", "rb") as model_file:
    svc_model = pickle.load(model_file)

# Streamlit app
st.title("Risk Classification Prediction")

st.sidebar.header("Input Features")


def user_input_features():
    industrial_risk = st.sidebar.slider("Industrial Risk", 0.0, 1.0, 0.5)
    management_risk = st.sidebar.slider("Management Risk", 0.0, 1.0, 1.0)
    financial_flexibility = st.sidebar.slider("Financial Flexibility", 0.0, 1.0, 0.0)
    credibility = st.sidebar.slider("Credibility", 0.0, 1.0, 0.0)
    competitiveness = st.sidebar.slider("Competitiveness", 0.0, 1.0, 0.0)
    operating_risk = st.sidebar.slider("Operating Risk", 0.0, 1.0, 0.5)

    features = np.array([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness,
                         operating_risk]).reshape(1, -1)
    return features


features = user_input_features()

if st.button("Predict"):
    prediction = svc_model.predict(features)
    st.write(f"Predicted Class: {int(prediction[0])}")
