import streamlit as st
import pandas as pd
import joblib

# 1. Carregar modelo e scaler
modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

# 2. Configuração da página
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

# 3. Formulário (apenas com as 24 features originais)
with st.form("formulario"):
    # Dados numéricos
    age = st.slider("Idade", 1, 120, 45)
    bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=28.5)
    glucose = st.number_input("Glicemia em jejum (mg/dL)", min_value=50, max_value=300, value=100)
    hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.5)
    
    # Dados categóricos
    gender = st.selectbox("Sexo", ["Feminino", "Masculino"])
    hypertension = st.selectbox("Hipertensão", ["Não", "Sim"])
    heart_disease = st.selectbox("Doença Cardíaca", ["Não", "Sim"])
    smoking = st.selectbox("Tabagismo", ["Nunca fumou", "Fumava anteriormente", "Fuma atualmente"])
    
    submit = st.form_submit_button("Prever")

# 4. Codificação CORRETA (garantindo 24 features)
if submit:
    entrada = {
        # Features numéricas
        "Age": age,
        "BMI": bmi,
        "Fasting_Blood_Glucose": glucose,
        "HbA1c": hba1c,
        # Features categóricas codificadas
        "Sex_Male": 1 if gender == "Masculino" else 0,
        "Hypertension_Yes": 1 if hypertension == "Sim" else 0,
        "Heart_Disease_Yes": 1 if heart_disease == "Sim" else 0,
        "Smoking_Status_Former": 1 if smoking == "Fumava anteriormente" else 0,
        "Smoking_Status_Never": 1 if smoking == "Nunca fumou" else 0,
        # Preencher outras features com valores padrão (ajuste conforme seu modelo)
        "Waist_Circumference": 90,  # valor médio
        "Blood_Pressure_Systolic": 120,
        "Blood_Pressure_Diastolic": 80,
        # ... (adicionar todas as 24 features listadas no scaler)
    }

    # 5. Criar DataFrame com a ORDEM CORRETA
    dados = pd.DataFrame([entrada])[scaler.feature_names_in_]
    
    # 6. Verificação final (debug)
    st.write("### Verificação")
    st.write(f"Número de features: {dados.shape[1]}")
    st.write("Colunas:", dados.columns.tolist())

    # 7. Predição
    try:
        dados_normalizados = scaler.transform(dados)
        resultado = modelo.predict(dados_normalizados)[0]
        st.success("✅ Sem diabetes" if resultado == 0 else "⚠️ Risco de diabetes")
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        st.write("Dados problemáticos:", dados)
