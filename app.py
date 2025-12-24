import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle

st.set_page_config(page_title='Customer Analytics', layout='wide')

@st.cache_resource
def load_models():
    m1 = tf.keras.models.load_model('classificationmodel.h5', compile=False)
    m1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m2 = tf.keras.models.load_model('regressionmodel.h5', compile=False)
    m2.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
    return m1, m2

@st.cache_resource
def load_preprocessing():
    with open('label_encoder_gender.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('one_hot_encoder_geo.pkl', 'rb') as f:
        ohe = pickle.load(f)
    with open('class_scaler.pkl', 'rb') as f:
        c_sc = pickle.load(f)
    with open('reg_scaler.pkl', 'rb') as f:
        r_sc = pickle.load(f)
    return le, ohe, c_sc, r_sc

model_churn, model_salary = load_models()
label_encoder, ohe_encoder, class_scaler, reg_scaler = load_preprocessing()

nav_col, _ = st.columns([1, 2]) 
with nav_col:
    app_mode = st.radio("📍 Select Task", 
                        ["Churn Prediction", "Salary Prediction"], 
                        horizontal=True)

if app_mode == "Churn Prediction":
    st.title('💳 Customer Churn Prediction')
    st.write("Using ANN to predict if a customer will leave the bank.")
else:
    st.title('💰 Customer Salary Prediction')
    st.write("Using ANN to estimate a customer's salary.")

st.divider()


col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', ohe_encoder.categories_[0])
    gender = st.selectbox('Gender', label_encoder.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance', min_value=0.0)
    credit_score = st.slider('Credit Score', 300, 900)

with col2:
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_mem = st.selectbox('Is an Active Member', [0, 1])
    
    if app_mode == "Churn Prediction":
        extra_val = st.number_input('Estimated Salary', min_value=0.0)
        extra_col_name = 'EstimatedSalary'
    else:
        extra_val = st.selectbox('Has Exited?', [0, 1])
        extra_col_name = 'Exited'

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_mem],
    extra_col_name: [extra_val]
})

geo_encoded = ohe_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


if st.button('Predict'):
    if app_mode == "Churn Prediction":
        input_data_scaled = class_scaler.transform(input_data)
        prediction = model_churn.predict(input_data_scaled)
        prob = prediction[0][0]
        st.subheader(f'Churn Probability: {prob:.2%}')
        if prob > 0.5:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is likely to stay.')
    else:
        input_data_scaled = reg_scaler.transform(input_data)
        prediction = model_salary.predict(input_data_scaled)
        st.subheader(f'Estimated Salary: ${prediction[0][0]:,.2f}')