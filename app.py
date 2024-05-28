# app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
clf = joblib.load('heart_model.pkl')

st.title("Heart Disease Prediction")

st.write("""
# Heart Disease Prediction App
This app predicts whether a person has heart disease or not!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 44)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.slider('Chest Pain Type (cp)', 0, 3, 0)
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (chol)', 126, 564, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar (fbs)', ['<120 mg/dl', '>120 mg/dl'])
    restecg = st.sidebar.slider('Resting Electrocardiographic Results (restecg)', 0, 2, 0)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ['Yes', 'No'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest (oldpeak)', 0.0, 6.2, 0.0)
    slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment (slope)', 0, 2, 0)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Flouroscopy (ca)', 0, 4, 0)
    thal = st.sidebar.slider('Thalassemia (thal)', 0, 3, 0)
    
    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == '>120 mg/dl' else 0
    exang = 1 if exang == 'Yes' else 0
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = np.array(list(data.values())).reshape(1, -1)
    return features

input_features = user_input_features()

st.subheader('User Input parameters')
st.write(input_features)

# Predict using the input features
prediction = clf.predict(input_features)

st.subheader('Prediction')
st.write(prediction)
