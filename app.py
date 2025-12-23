import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = tf.keras.models.load_model('model.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    ohe_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title='Churn Predictor', layout='wide')

st.title('💳 Customer Churn Prediction')
st.write("**What it does:** The ANN-based model predicts customer churn using their demographic and financial data.")
st.divider()


col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', ohe_encoder.categories_[0])
    gender = st.selectbox('Gender', label_encoder.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.slider('Credit Score', min_value=300, max_value=900)

with col2:
    estimated_salary = st.number_input('Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_mem = st.selectbox('Is an Active Member', [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_mem],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = ohe_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    prediction_probab = prediction[0][0]

    st.write('The probability to churn is ', prediction_probab)
    if prediction_probab > 0.5:
        st.write('The customer is likely to churn')
    else:
        st.write('The customer is not likely to churn')