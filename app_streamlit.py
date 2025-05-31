import streamlit as st
import pandas as pd
import joblib

# Carregar modelo e scaler
modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

# Entradas do usuário
age = st.slider("Idade", 1, 120, 45)
bmi = st.number_input("IMC", 10.0, 60.0, 28.5)
waist = st.number_input("Cintura (cm)", 50.0, 200.0, 90.0)
glucose = st.number_input("Glicose jejum", 50, 300, 100)
hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.8)
hdl = st.number_input("Colesterol HDL", 20.0, 100.0, 55.0)
ldl = st.number_input("Colesterol LDL", 30.0, 200.0, 110.0)
chol_total = st.number_input("Colesterol Total", 100.0, 300.0, 190.0)
ggt = st.number_input("GGT", 10, 100, 30)
urate = st.number_input("Ácido Úrico (Serum Urate)", 1.0, 10.0, 5.2)
calories = st.number_input("Calorias ingeridas", 1000, 5000, 2200)
bp_sys = st.number_input("Pressão Sistólica", 80, 200, 120)
bp_dia = st.number_input("Pressão Diastólica", 40, 130, 75)


# Sexo
gender = st.selectbox("Sexo", ["Masculino", "Feminino"])
sexo = 1 if gender == "Masculino" else 0

# Criar dicionário com os dados
entrada = {
    "Age": age,
    "BMI": bmi,
    "Waist_Circumference": waist,
    "Fasting_Blood_Glucose": glucose,
    "Blood_Pressure_Systolic": bp_sys,
    "Blood_Pressure_Diastolic": bp_dia,
    "Cholesterol_Total": chol_total,
    "Cholesterol_HDL": hdl,
    "Cholesterol_LDL": ldl,
    "GGT": ggt,
    "Serum_Urate": urate,
    "Dietary_Intake_Calories": calories,
    "Family_History_of_Diabetes": 1,
    "Previous_Gestational_Diabetes": 0,
    "Sex_Male": 1 if gender == "Masculino" else 0,
    "Ethnicity_White": 1,
    "Ethnicity_Black": 0,
    "Ethnicity_Hispanic": 0,
    "Physical_Activity_Level_Low": 0,
    "Physical_Activity_Level_Moderate": 1,
    "Alcohol_Consumption_None": 1,
    "Alcohol_Consumption_Moderate": 0,
    "Smoking_Status_Former": 0,
    "Smoking_Status_Never": 1,
    "HbA1c": hba1c,
}


df = pd.DataFrame([entrada])

# Ordem correta das 24 colunas
colunas_ordenadas = [
    "Age", "BMI", "Waist_Circumference", "Fasting_Blood_Glucose", 
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
    "Cholesterol_Total", "Cholesterol_HDL", "Cholesterol_LDL", "GGT",
    "Serum_Urate", "Dietary_Intake_Calories", "Family_History_of_Diabetes",
    "Previous_Gestational_Diabetes", "Sex_Male", 
    "Ethnicity_White", "Ethnicity_Black", "Ethnicity_Hispanic",
    "Physical_Activity_Level_Low", "Physical_Activity_Level_Moderate",
    "Alcohol_Consumption_None", "Alcohol_Consumption_Moderate",
    "Smoking_Status_Former", "Smoking_Status_Never","HbA1c"
]


# Reordenar e normalizar
df = df.reindex(columns=colunas_ordenadas)

# Verificação
st.subheader("🔎 Verificação")
st.write("Colunas enviadas:", df.columns.tolist())
st.write("Shape:", df.shape)

# Predição
try:
    dados_normalizados = scaler.transform(df)
    if st.button("🔍 Prever"):
        pred = modelo.predict(dados_normalizados)[0]
        st.success("✅ Diabetes detectado!" if pred == 1 else "🟢 Sem sinais de diabetes.")
except Exception as e:
    st.error(f"Erro na predição: {e}")

