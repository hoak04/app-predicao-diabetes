
import streamlit as st
import pandas as pd
import joblib

# Carregar modelo e scaler
modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")
st.write("Preencha os dados abaixo para prever a presença de diabetes:")

# Formulário
age = st.slider("Idade", 1, 120, 45)
bmi = st.number_input("IMC (Índice de Massa Corporal)", min_value=10.0, max_value=60.0, value=28.5)
hba1c = st.number_input("Hemoglobina glicada (HbA1c)", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Glicemia em jejum (mg/dL)", min_value=50, max_value=300, value=100)

gender = st.selectbox("Sexo", ["Masculino", "Feminino"])
hypertension = st.selectbox("Hipertensão", ["Sim", "Não"])
heart_disease = st.selectbox("Doença Cardíaca", ["Sim", "Não"])
smoking = st.selectbox("Histórico de Tabagismo", ["Nunca fumou", "Fuma atualmente", "Fumava anteriormente"])

# Codificação
genero_m = 1 if gender == "Masculino" else 0
hipertensao = 1 if hypertension == "Sim" else 0
doenca_cardiaca = 1 if heart_disease == "Sim" else 0
smoking_history = {
    "Nunca fumou": [0, 0],
    "Fumava anteriormente": [1, 0],
    "Fuma atualmente": [0, 1]
}
smoke_encoded = smoking_history[smoking]

smoke_current = 1 if smoking == "Fuma atualmente" else 0
smoke_former = 1 if smoking == "Fumava anteriormente" else 0
smoke_never = 1 if smoking == "Nunca fumou" else 0

# Dados do usuário

entrada = {
    "Age": age,
    "BMI": bmi,
    "HbA1c": hba1c,
    "Fasting_Blood_Glucose": glucose,
    "Gender_Male": genero_m,
    "Hypertension_Yes": hipertensao,
    "Heart_Disease_Yes": doenca_cardiaca,
    "Smoking_History_current": smoke_current,
    "Smoking_History_former": smoke_former,
    "Smoking_History_never": smoke_never
}

# Criar DataFrame e alinhar com as colunas do scaler
dados = pd.DataFrame([entrada])
colunas_esperadas = list(scaler.feature_names_in_)
dados = dados.reindex(columns=colunas_esperadas, fill_value=0)
st.write("Features esperadas pelo modelo:", modelo.feature_names_in_)
st.write("Features que estamos enviando:", dados.columns.tolist())
st.write("Colunas finais:", dados.columns.tolist())
st.write("Shape final:", dados.shape)
dados_normalizados = scaler.transform(dados.to_numpy())


# Predição

if st.button("🔍 Prever"):
    try:
        dados_normalizados = scaler.transform(dados)
        resultado = modelo.predict(dados_normalizados)[0]
        st.success("✅ Resultado: **Diabetes detectado!**" if resultado == 1 else "🟢 Resultado: **Sem sinais de diabetes.**")
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")
        st.write("Dados enviados:", dados)
        st.write("Shape dos dados:", dados.shape)
