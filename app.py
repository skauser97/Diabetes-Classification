import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_recall_fscore_support, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

## load pickle file and model_selection

## load pickle file and model_selection

with open("pickle/robust_scaler.pkl", "rb") as f:
    transformer = pickle.load(f)

with open("pickle/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("pickle/onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

with open("models/model_1.pkl", "rb") as file:
    model_1 = pickle.load(file)

def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi <= 25:
        return "Normal"
    elif bmi <= 30:
        return "Overweight"
    elif bmi <= 35:
        return "Obesity I"
    elif bmi <= 40:
        return "Obesity II"
    else:
        return "Obesity III"

def categorize_glucose(glucose):
    if glucose < 70:
        return "Low Glucose"
    elif glucose <= 99:
        return "Normal"
    elif glucose <= 125:
        return "Prediabetic"
    else:
        return "High Glucose"

def insulin_score(insulin):
    if 16 <= insulin <= 165:
        return "Normal"
    else:
        return "Abnormal"

## streamlit app

st.title("Diabetes classification")
# user input
age = st.slider('Age', 18,90)
Pregnancies=st.number_input('Pregnancies')
Glucose=st.number_input('Glucose')
BloodPressure=st.number_input('BloodPressure')
SkinThickness=st.number_input('SkinThickness')
Insulin=st.number_input('Insulin')
BMI=st.number_input('BMI')
DiabetesPedigreeFunction=st.number_input('DiabetesPedigreeFunction')

input_df = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [age]
})

if st.button("Predict Diabetes Status"):
    try:
        input_df = pd.DataFrame({
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BloodPressure': [BloodPressure],
            'SkinThickness': [SkinThickness],
            'Insulin': [Insulin],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [age]
        })


        input_df["Insulin"] = input_df["Insulin"].apply(insulin_score)
        input_df["Glucose"] = input_df["Glucose"].apply(categorize_glucose)
        input_df["BMI"] = input_df["BMI"].apply(categorize_bmi)

        input_df["Insulin"] = label_encoder.transform(input_df["Insulin"])
        input_cat = onehot_encoder.transform(input_df[["BMI","Glucose"]])
        input_cat_columns = onehot_encoder.get_feature_names_out(["BMI", "Glucose"])
        #converting into a adataframe
        input_cat_encoded = pd.DataFrame(input_cat.toarray(),columns= input_cat_columns)
        input_df= input_df.drop(['Glucose','BMI'], axis =1 )
        cols = input_df.columns
        input_df_index = input_df.index

        input_df_scaled=transformer.transform(input_df)
        input_scaled=pd.DataFrame(input_df_scaled, columns = cols, index = input_df_index)
        #input_final = pd.concat([input_scaled, input_cat_encoded], axis=1)
        input_final = pd.concat([input_scaled.reset_index(drop=True),
                                 input_cat_encoded.reset_index(drop=True)], axis=1)
        

        #X_test_concat = pd.concat([X_test, X_test_encoded], axis=1)


        

        prediction = model_1.predict(input_final)[0]

        if prediction == 1:
            st.success("Prediction: The person **has diabetes**.")
        else:
            st.info("Prediction: The person **does not have diabetes**.")
    
    except Exception as e:
        st.error(f"Error: {e}")


