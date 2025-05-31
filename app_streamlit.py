import streamlit as st
import pandas as pd
import joblib

# Carrega o modelo e o scaler
modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

# Configuração da página
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

# Inputs do usuário
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

# Dicionário de entrada (garanta que todas as features do modelo estão aqui)
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
    "Family_History_of_Diabetes": 1,  # Exemplo (ajuste conforme necessário)
    "Previous_Gestational_Diabetes": 0,
    "Sex_Male": 1,  # 1 para masculino, 0 para feminino
    "Ethnicity_White": 1,  # Ajuste para outras etnias se necessário
    "Ethnicity_Black": 0,
    "Ethnicity_Hispanic": 0,
    "Physical_Activity_Level_Low": 0,
    "Physical_Activity_Level_Moderate": 1,
    "Alcohol_Consumption_None": 1,
    "Alcohol_Consumption_Moderate": 0,
    "Smoking_Status_Former": 0,
    "Smoking_Status_Never": 1,
    "HbA1c": hba1c
}

# Cria o DataFrame
df = pd.DataFrame([entrada])

# Verifica se todas as colunas esperadas estão presentes
try:
    # Normaliza os dados
    dados_normalizados = scaler.transform(df)
    
    if st.button("🔍 Prever"):
        pred = modelo.predict(dados_normalizados)[0]
        st.success("✅ Diabetes detectado!" if pred == 1 else "🟢 Sem sinais de diabetes.")
except Exception as e:
    st.error(f"Erro: {e}")
    st.write("🔴 Verifique se todas as features do modelo estão presentes e na ordem correta.")
    st.write("Features esperadas:", modelo.feature_names_in_)  # Se o modelo for scikit-learn >= 1.0
