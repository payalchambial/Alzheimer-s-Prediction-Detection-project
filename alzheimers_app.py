# alzheimers_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc

# Load and clean dataset
data = pd.read_csv(r"C:\Users\pc\Documents\Alzhemier.csv")
data.fillna(data.mean(), inplace=True)

# Encode Gender if needed
encoder = LabelEncoder()
if data['Gender'].dtype == 'object':
    data['Gender'] = encoder.fit_transform(data['Gender'])

# Features and target
X = data.drop('Alzheimer', axis=1)
y = data['Alzheimer']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Test accuracy
y_pred = lr_model.predict(X_test)
st.write("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# App Title
st.title("Alzheimer's Prediction App")
st.header("Enter Patient Details:")

# User inputs
age = st.number_input("Age", min_value=30, max_value=100, value=65)
gender = st.selectbox("Gender", ["Male", "Female"])
mmse = st.number_input("MMSE Score", min_value=0.0, max_value=30.0, value=23.0)
brain_volume = st.number_input("Brain Volume", min_value=0.0, max_value=1.0, value=0.72)

# Encode gender
gender_encoded = 1 if gender == "Male" else 0

# Prepare data for prediction
new_patient = np.array([[age, gender_encoded, mmse, brain_volume]])
new_patient_scaled = scaler.transform(new_patient)

# Prediction button
if st.button("Predict"):
    prediction = lr_model.predict(new_patient_scaled)
    if prediction[0] == 1:
        st.error("Alzheimer Detected (Early Stage)")
    else:
        st.success("No Alzheimer Detected")

# MRI Image Section
st.header(" Brain MRI Image")

uploaded_file = st.file_uploader("Upload a brain MRI image (optional)", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_mri = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(uploaded_mri, caption="Uploaded MRI", width=300, clamp=True)
    
    # Extract features from MRI
    # Example: use mean intensity as brain_volume
    brain_volume_from_mri = uploaded_mri.mean() / 255
    st.write(f"Extracted Brain Volume from MRI: {brain_volume_from_mri:.2f}")
    
    # Optionally, add more MRI features
    std_intensity = uploaded_mri.std() / 255
    edges = cv2.Canny(uploaded_mri, 100, 200)
    edge_density = edges.sum() / (uploaded_mri.shape[0] * uploaded_mri.shape[1])
    
    # Update patient input for prediction
    new_patient = np.array([[age, gender_encoded, mmse, brain_volume_from_mri]])
    new_patient_scaled = scaler.transform(new_patient)

# Dataset and Plots
if st.checkbox("Show Dataset"):
    st.write(data.head())

if st.checkbox("Show Alzheimer Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(x='Alzheimer', data=data, ax=ax)
    st.pyplot(fig)
    
if st.checkbox("Show ROC Curve with Patient"):
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Patient probability
    patient_prob = lr_model.predict_proba(new_patient_scaled)[0][1]
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0,1], [0,1], linestyle='--', color='gray')
    
    # Highlight patient
    ax.scatter(1-patient_prob, patient_prob, color='red', s=100, label="New Patient")
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)