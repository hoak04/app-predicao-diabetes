import streamlit as st
import pandas as pd
import joblib

# Carregar modelo e scaler
modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

# Configuração da página
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")
st.write("Preencha os dados disponíveis abaixo:")

# --- Seção 1: Dados Básicos (já existentes) ---
st.header("Dados Pessoais")
age = st.slider("Idade", 1, 120, 45)
gender = st.selectbox("Sexo", ["Masculino", "Feminino"])
bmi = st.number_input("IMC (Índice de Massa Corporal)", min_value=10.0, max_value=60.0, value=28.5)
waist_circumference = st.number_input("Circunferência da Cintura (cm)", min_value=50, max_value=200, value=90)

# --- Seção 2: Dados Clínicos ---
st.header("Dados Clínicos")
glucose = st.number_input("Glicemia em jejum (mg/dL)", min_value=50, max_value=300, value=100)
hba1c = st.number_input("Hemoglobina glicada (HbA1c)", min_value=3.0, max_value=15.0, value=5.5)
bp_systolic = st.number_input("Pressão Arterial Sistólica", min_value=80, max_value=200, value=120)
bp_diastolic = st.number_input("Pressão Arterial Diastólica", min_value=50, max_value=120, value=80)

# --- Seção 3: Dados de Saúde (valores padrão para features não coletadas) ---
st.header("Dados Adicionais (Preenchidos Automaticamente)")
st.warning("Os campos abaixo contêm valores médios estimados. Altere se souber o valor real.")

# Valores médios razoáveis (ajuste conforme necessário)
cholesterol_total = st.number_input("Colesterol Total (mg/dL)", min_value=100, max_value=400, value=200, disabled=True)
cholesterol_hdl = st.number_input("Colesterol HDL (mg/dL)", min_value=20, max_value=100, value=50, disabled=True)
cholesterol_ldl = st.number_input("Colesterol LDL (mg/dL)", min_value=50, max_value=300, value=100, disabled=True)
ggt = st.number_input("GGT (U/L)", min_value=5, max_value=200, value=25, disabled=True)
serum_urate = st.number_input("Ácido Úrico (mg/dL)", min_value=2.0, max_value=10.0, value=5.0, disabled=True)
calories = st.number_input("Ingestão Calórica Diária", min_value=1000, max_value=5000, value=2000, disabled=True)

# --- Seção 4: Históricos ---
st.header("Históricos")
family_history = st.selectbox("Histórico Familiar de Diabetes", ["Não", "Sim"])
gestational_diabetes = st.selectbox("Diabetes Gestacional Prévio (se aplicável)", ["Não", "Sim"])
hypertension = st.selectbox("Hipertensão", ["Não", "Sim"])
heart_disease = st.selectbox("Doença Cardíaca", ["Não", "Sim"])
smoking = st.selectbox("Tabagismo", ["Nunca fumou", "Fumava anteriormente", "Fuma atualmente"])
ethnicity = st.selectbox("Etnia", ["Branca", "Negra", "Hispânica", "Outra"])
physical_activity = st.selectbox("Nível de Atividade Física", ["Baixo", "Moderado", "Alto"])
alcohol = st.selectbox("Consumo de Álcool", ["Nenhum", "Moderado", "Excessivo"])

# --- Codificação das Variáveis ---
entrada = {
    # Dados numéricos
    "Age": age,
    "BMI": bmi,
    "Waist_Circumference": waist_circumference,
    "Fasting_Blood_Glucose": glucose,
    "HbA1c": hba1c,
    "Blood_Pressure_Systolic": bp_systolic,
    "Blood_Pressure_Diastolic": bp_diastolic,
    "Cholesterol_Total": cholesterol_total,
    "Cholesterol_HDL": cholesterol_hdl,
    "Cholesterol_LDL": cholesterol_ldl,
    "GGT": ggt,
    "Serum_Urate": serum_urate,
    "Dietary_Intake_Calories": calories,
    
    # Dados categóricos codificados
    "Family_History_of_Diabetes": 1 if family_history == "Sim" else 0,
    "Previous_Gestational_Diabetes": 1 if gestational_diabetes == "Sim" else 0,
    "Sex_Male": 1 if gender == "Masculino" else 0,
    "Ethnicity_Black": 1 if ethnicity == "Negra" else 0,
    "Ethnicity_Hispanic": 1 if ethnicity == "Hispânica" else 0,
    "Ethnicity_White": 1 if ethnicity == "Branca" else 0,
    "Physical_Activity_Level_Low": 1 if physical_activity == "Baixo" else 0,
    "Physical_Activity_Level_Moderate": 1 if physical_activity == "Moderado" else 0,
    "Alcohol_Consumption_Moderate": 1 if alcohol == "Moderado" else 0,
    "Alcohol_Consumption_None": 1 if alcohol == "Nenhum" else 0,
    "Smoking_Status_Former": 1 if smoking == "Fumava anteriormente" else 0,
    "Smoking_Status_Never": 1 if smoking == "Nunca fumou" else 0,
    
    # Variáveis omitidas no formulário (preenchidas com 0)
    "Hypertension_Yes": 1 if hypertension == "Sim" else 0,
    "Heart_Disease_Yes": 1 if heart_disease == "Sim" else 0
}

# --- Garantir a ordem correta das colunas ---
dados = pd.DataFrame([entrada])[scaler.feature_names_in_]

# --- Debug (opcional) ---
with st.expander("🔍 Ver dados enviados ao modelo"):
    st.write("Colunas:", dados.columns.tolist())
    st.write("Valores:", dados.iloc[0].to_dict())

# --- Predição ---
if st.button("🔍 Prever Diabetes"):
    try:
        dados_normalizados = scaler.transform(dados)
        resultado = modelo.predict(dados_normalizados)[0]
        
        if resultado == 1:
            st.error("## Resultado: Risco de Diabetes Detectado")
            st.warning("Consulte um médico para avaliação detalhada.")
        else:
            st.success("## Resultado: Sem sinais de diabetes")
            st.info("Continue mantendo hábitos saudáveis!")
            
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        st.write("Dados enviados:", dados)
