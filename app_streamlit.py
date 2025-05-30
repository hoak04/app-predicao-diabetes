import streamlit as st
import pandas as pd
import joblib

# Carregar modelo e scaler
try:
    modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
    scaler = joblib.load("modelo/scaler.pkl")
except Exception as e:
    st.error(f"Erro ao carregar modelo: {str(e)}")
    st.stop()

# Configuração da página
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

# Verificar features esperadas
if not hasattr(scaler, 'feature_names_in_'):
    st.error("O scaler não contém informações das features originais!")
    st.stop()

colunas_esperadas = scaler.feature_names_in_.tolist()
st.sidebar.write("### Features esperadas pelo modelo:")
st.sidebar.write(colunas_esperadas)

# Formulário de entrada
with st.form("form_diabetes"):
    st.header("Informações Básicas")
    age = st.slider("Idade", 1, 120, 45)
    bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=28.5)
    gender = st.selectbox("Sexo", ["Feminino", "Masculino"])
    
    st.header("Indicadores Clínicos")
    glucose = st.number_input("Glicemia em jejum (mg/dL)", min_value=50, max_value=300, value=100)
    hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
    waist_circumference = st.number_input("Circunferência da cintura (cm)", min_value=50, max_value=200, value=90)
    
    st.header("Histórico de Saúde")
    hypertension = st.selectbox("Hipertensão", ["Não", "Sim"])
    heart_disease = st.selectbox("Doença Cardíaca", ["Não", "Sim"])
    smoking = st.selectbox("Tabagismo", ["Nunca fumou", "Fumava anteriormente", "Fuma atualmente"])
    
    submit = st.form_submit_button("Realizar Predição")

# Processamento após submissão
if submit:
    # 1. Codificação CORRETA das variáveis (com nomes exatos)
    entrada = {
        # Features numéricas (nomes exatos como no scaler)
        "Age": age,
        "BMI": bmi,
        "Waist_Circumference": waist_circumference,
        "Fasting_Blood_Glucose": glucose,  # Nome corrigido (sem typo)
        "HbA1c": hba1c,
        
        # Features categóricas codificadas
        "Sex_Male": 1 if gender == "Masculino" else 0,
        "Hypertension_Yes": 1 if hypertension == "Sim" else 0,
        "Heart_Disease_Yes": 1 if heart_disease == "Sim" else 0,
        "Smoking_Status_Former": 1 if smoking == "Fumava anteriormente" else 0,
        "Smoking_Status_Never": 1 if smoking == "Nunca fumou" else 0,
        
        # Preencher outras features com valores padrão
        "Blood_Pressure_Systolic": 120,
        "Blood_Pressure_Diastolic": 80,
        "Cholesterol_Total": 200,
        "Cholesterol_HDL": 50,
        "Cholesterol_LDL": 100,
        "GGT": 25,
        "Serum_Urate": 5.0,
        "Dietary_Intake_Calories": 2000,
        "Family_History_of_Diabetes": 0,
        "Previous_Gestational_Diabetes": 0,
        "Ethnicity_Black": 0,
        "Ethnicity_Hispanic": 0,
        "Ethnicity_White": 1,
        "Physical_Activity_Level_Low": 1,  # Nome corrigido (Level com L maiúsculo)
        "Physical_Activity_Level_Moderate": 0,
        "Alcohol_Consumption_Moderate": 0,
        "Alcohol_Consumption_None": 1
    }

    # 2. Criar DataFrame garantindo a ordem correta
    dados = pd.DataFrame([entrada])[colunas_esperadas]
    
    # 3. Verificação final
    st.write("### Dados que serão processados:")
    st.write(dados)
    
    # 4. Validação crítica
    if list(dados.columns) != list(scaler.feature_names_in_):
        st.error("ERRO: Incompatibilidade nas colunas!")
        st.write("Diferenças:")
        st.write("Enviadas:", dados.columns.tolist())
        st.write("Esperadas:", scaler.feature_names_in_.tolist())
        st.stop()
    
    # 5. Predição
    try:
        dados_normalizados = scaler.transform(dados)
        resultado = modelo.predict(dados_normalizados)[0]
        
        if resultado == 1:
            st.error("## Resultado: Risco de Diabetes Detectado")
            st.warning("Recomenda-se consulta médica para avaliação detalhada.")
        else:
            st.success("## Resultado: Sem sinais de diabetes")
            st.info("Continue mantendo hábitos saudáveis!")
            
    except Exception as e:
        st.error(f"Erro durante a predição: {str(e)}")
        st.write("Detalhes técnicos para diagnóstico:")
        st.write("Shape dos dados:", dados.shape)
        st.write("Tipo de dados:", type(dados))
