import streamlit as st
import pandas as pd
import joblib

# Carrega o modelo e o scaler
modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

# Inputs do usuário
age = st.slider("Idade", 1, 120, 45)
sex = st.radio("Sexo", ["Masculino", "Feminino"], index=0)
ethnicity = st.radio("Etnia", ["White", "Black", "Hispanic"], index=0)
bmi = st.number_input("IMC", 10.0, 60.0, 28.5)
waist = st.number_input("Cintura (cm)", 50.0, 200.0, 90.0)
glucose = st.number_input("Glicose jejum", 50, 300, 100)
hba1c = st.number_input("HbA1c", 3.0, 15.0, 5.8)
bp_sys = st.number_input("Pressão Sistólica", 80, 200, 120)
bp_dia = st.number_input("Pressão Diastólica", 40, 130, 75)
chol_total = st.number_input("Colesterol Total", 100.0, 300.0, 190.0)
hdl = st.number_input("Colesterol HDL", 20.0, 100.0, 55.0)
ldl = st.number_input("Colesterol LDL", 30.0, 200.0, 110.0)
ggt = st.number_input("GGT", 10, 100, 30)
urate = st.number_input("Ácido Úrico (Serum Urate)", 1.0, 10.0, 5.2)
physical_activity = st.radio("Nível de Atividade Física", ["Low", "Moderate", "High"], index=1)
calories = st.number_input("Calorias ingeridas", 1000, 5000, 2200)
alcohol = st.radio("Consumo de Álcool", ["None", "Moderate", "Heavy"], index=0)
smoking = st.radio("Status de Fumante", ["Never", "Former", "Current"], index=0)
family_history = st.checkbox("Histórico Familiar de Diabetes", value=True)
gestational_diabetes = st.checkbox("Diabetes Gestacional Prévia", value=False)

# Pré-processamento das variáveis categóricas
sex_male = 1 if sex == "Masculino" else 0

ethnicity_white = 1 if ethnicity == "White" else 0
ethnicity_black = 1 if ethnicity == "Black" else 0
ethnicity_hispanic = 1 if ethnicity == "Hispanic" else 0

physical_activity_low = 1 if physical_activity == "Low" else 0
physical_activity_moderate = 1 if physical_activity == "Moderate" else 0

alcohol_none = 1 if alcohol == "None" else 0
alcohol_moderate = 1 if alcohol == "Moderate" else 0

smoking_never = 1 if smoking == "Never" else 0
smoking_former = 1 if smoking == "Former" else 0

# Dicionário de entrada na ORDEM CORRETA
entrada = {
    "Age": age,
    "Sex": 1 if sex == "Masculino" else 0  # Assumindo que Sex_Male já foi codificado (1 para masculino)
    "Ethnicity_White": 1 if ethnicity == "White" else 0,  # Corrigido de "Ethnicity_Write"
    "Ethnicity_Black": 1 if ethnicity == "Black" else 0,
    "Ethnicity_Hispanic": 1 if ethnicity == "Hispanic" else 0,
    "BMI": bmi,
    "Waist_Circumference": waist,
    "Fasting_Blood_Glucose": glucose,
    "HbA1c": hba1c,
    "Blood_Pressure_Systolic": bp_sys,
    "Blood_Pressure_Diastolic": bp_dia,
    "Cholesterol_Total": chol_total,
    "Cholesterol_HDL": hdl,
    "Cholesterol_LDL": ldl,
    "GGT": ggt,
    "Serum_Urate": urate,
    "Physical_Activity_Level_Low": 1 if physical_activity == "Low" else 0,
    "Physical_Activity_Level_Moderate": 1 if physical_activity == "Moderate" else 0,
    "Dietary_Intake_Calories": calories,
    "Alcohol_Consumption_None": 1 if alcohol == "None" else 0,
    "Alcohol_Consumption_Moderate": 1 if alcohol == "Moderate" else 0,
     "Smoking_Status_Never": 1 if smoking == "Never" else 0,  
    "Smoking_Status_Former": 1 if smoking == "Former" else 0,
    "Family_History_of_Diabetes": 1 if family_history else 0,
    "Previous_Gestational_Diabetes": 1 if gestational_diabetes else 0
}

# Ordem EXATA das features conforme especificado
ordem_features = [
    "Age",
    "Sex",
    "Ethnicity_White",
    "Ethnicity_Black",
    "Ethnicity_Hispanic",
    "BMI",
    "Waist_Circumference",
    "Fasting_Blood_Glucose",
    "HbA1c",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Cholesterol_Total",
    "Cholesterol_HDL",
    "Cholesterol_LDL",
    "GGT",
    "Serum_Urate",
    "Physical_Activity_Level_Low",
    "Physical_Activity_Level_Moderate",
    "Dietary_Intake_Calories",
    "Alcohol_Consumption_None",
    "Alcohol_Consumption_Moderate",
    "Smoking_Status_Never",
    "Smoking_Status_Former",
    "Family_History_of_Diabetes",
    "Previous_Gestational_Diabetes"
]

# Cria o DataFrame na ordem correta
df = pd.DataFrame([entrada])[ordem_features]

# Verificação (opcional)
st.subheader("🔎 Verificação")
st.write("Features enviadas:", df.columns.tolist())
st.write("Valores:", df.values.tolist()[0])

st.write("### 🔍 Verificação de Features")
st.write("Features que estou enviando:", df.columns.tolist())

if hasattr(modelo, 'feature_names_in_'):
    st.write("Features que o modelo espera:", modelo.feature_names_in_)
    
    # Encontra discrepâncias
    missing = set(modelo.feature_names_in_) - set(df.columns)
    extra = set(df.columns) - set(modelo.feature_names_in_)
    
    if missing:
        st.error(f"🚨 Features FALTANDO: {list(missing)}")
    if extra:
        st.warning(f"⚠️ Features EXTRAS: {list(extra)}")

if hasattr(modelo, 'feature_names_in_'):
    # Garante a ordem e features corretas
    df = df.reindex(columns=modelo.feature_names_in_, fill_value=0)
# Previsão
try:
    dados_normalizados = scaler.transform(df)
    if st.button("🔍 Prever"):
        pred = modelo.predict(dados_normalizados)[0]
        st.success("✅ Diabetes detectado!" if pred == 1 else "🟢 Sem sinais de diabetes.")
except Exception as e:
    st.error(f"Erro na predição: {str(e)}")
    st.write("Verifique se todas as features estão corretas:", df.columns.tolist())
