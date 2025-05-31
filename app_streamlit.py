
import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

age = st.slider("Idade", 1, 120, 45)
bmi = st.number_input("IMC", 10.0, 60.0, 28.5)
waist = st.number_input("Cintura (cm)", 50.0, 200.0, 90.0)
glucose = st.number_input("Glicose jejum", 50, 300, 100)

# Exemplo simplificado de entrada
entrada = {
    "Age": age,
    "BMI": bmi,
    "Waist_Circumference": waist,
    "Fasting_Blood_Glucose": glucose,
    "Sex_Male": 1,
    "Alcohol_Consumption_None": 1,
    "Smoking_Status_Never": 1,
    "Physical_Activity_Level_Moderate": 1,
    "Family_History_of_Diabetes": 1,
    "HbA1c": 5.8,
    "Serum_Urate": 5.2,
    "Cholesterol_Total": 190,
    "Cholesterol_HDL": 55,
    "Cholesterol_LDL": 110,
    "GGT": 30,
    "Dietary_Intake_Calories": 2200,
    "Previous_Gestational_Diabetes": 0,
    "Ethnicity_White": 1,
    "Ethnicity_Black": 0,
    "Ethnicity_Hispanic": 0,
    "Alcohol_Consumption_Moderate": 0,
    "Smoking_Status_Former": 0,
    "Physical_Activity_Level_Low": 0,
    "Ethnicity_Asian": 0
}

df = pd.DataFrame([entrada])
try:
    colunas_esperadas = list(modelo.feature_names_in_)
except AttributeError:
    colunas_esperadas = [
        "Age", "BMI", "Waist_Circumference", "Fasting_Blood_Glucose", "Sex_Male",
        "Alcohol_Consumption_None", "Alcohol_Consumption_Moderate",
        "Smoking_Status_Never", "Smoking_Status_Former",
        "Physical_Activity_Level_Moderate", "Physical_Activity_Level_Low",
        "Family_History_of_Diabetes", "Previous_Gestational_Diabetes",
        "Cholesterol_Total", "Cholesterol_HDL", "Cholesterol_LDL", "HbA1c", "GGT",
        "Serum_Urate", "Dietary_Intake_Calories", "Ethnicity_White",
        "Ethnicity_Black", "Ethnicity_Hispanic", "Ethnicity_Asian"
    ]

df = df.reindex(columns=colunas_esperadas, fill_value=0)

st.subheader("🔎 Verificação")
st.write("Colunas esperadas:", colunas_esperadas)
st.write("Colunas enviadas:", df.columns.tolist())
st.write("Shape:", df.shape)

try:
    dados_normalizados = scaler.transform(df)
    if st.button("🔍 Prever"):
        pred = modelo.predict(dados_normalizados)[0]
        st.success("✅ Diabetes detectado!" if pred == 1 else "🟢 Sem sinais de diabetes.")
except Exception as e:
    st.error(f"Erro na predição: {e}")
