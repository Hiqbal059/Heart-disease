import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st

# Load your trained model (replace 'model.pkl' with your model's file name)
best_model = joblib.load('resource/model.pkl')

# Function to predict heart disease
def predict_heart_disease(features):
    features = np.array(features).reshape(1, -1)
    prediction = best_model.predict(features)
    probability = best_model.predict_proba(features)[0][1] * 100
    return probability

# Streamlit app layout
st.title("Heart Disease Predictor")

# Define the features
feature_names = ['Age', 
                 'Sex (0 for Female & 1 for Male)', 
                 'Chest Pain Type (0,1,2,3)', 
                 'Resting Blood Pressure', 
                 'Serum Cholesterol in mg/dl', 
                 'Fasting Blood Sugar > 120 mg/dl (0 for No, 1 for Yes)', 
                 'Resting Electrocardiographic Results (values 0,1,2)', 
                 'Maximum Heart Rate Achieved', 
                 'Exercise Induced Angina (0 for No, 1 for Yes)', 
                 'Oldpeak', 
                 'The Slope Of The Peak exercise ST segment', 
                 'Number of major vessels (0-3) colored by fluoroscopy', 
                 'Thal: 0 = normal, 1 = fixed defect, 2 = reversible defect']

# Collect user inputs
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, step=1.0)
    user_inputs.append(value)

# When user clicks the button
if st.button("Predict"):
    try:
        # Predict the probability
        probability = predict_heart_disease(user_inputs)
        st.success(f"Heart Disease Probability: {probability:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")