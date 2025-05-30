import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Carregar modelo e scaler
modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
scaler = joblib.load("modelo/scaler.pkl")

# 2. Configuração da página
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

# 3. Obter as colunas exatas que o modelo espera
colunas_esperadas = scaler.feature_names_in_.tolist()
st.sidebar.write("⚠️ O modelo espera exatamente estas colunas:")
st.sidebar.write(colunas_esperadas)

# 4. Formulário dinâmico
entrada = {}
with st.form("formulario"):
    st.header("Dados Pessoais")
    if 'Age' in colunas_esperadas:
        entrada['Age'] = st.slider("Idade", 1, 120, 45)
    if 'BMI' in colunas_esperadas:
        entrada['BMI'] = st.number_input("IMC", min_value=10.0, max_value=60.0, value=28.5)
    
    st.header("Dados Clínicos")
    if 'Fasting_Blood_Glucose' in colunas_esperadas:
        entrada['Fasting_Blood_Glucose'] = st.number_input("Glicemia em jejum (mg/dL)", min_value=50, max_value=300, value=100)
    if 'HbA1c' in colunas_esperadas:
        entrada['HbA1c'] = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.5)
    
    st.header("Variáveis Categóricas")
    if 'Sex_Male' in colunas_esperadas:
        gender = st.selectbox("Sexo", ["Feminino", "Masculino"])
        entrada['Sex_Male'] = 1 if gender == "Masculino" else 0
    
    submit = st.form_submit_button("Prever")

# 5. Preencher valores faltantes com zero
for col in colunas_esperadas:
    if col not in entrada:
        entrada[col] = 0  # Ou use valores médios apropriados

# 6. Criar DataFrame com ordem exata
dados = pd.DataFrame([entrada])[colunas_esperadas]

# 7. Verificação final
st.write("### Dados que serão enviados ao modelo:")
st.write(dados)

# 8. Predição
if submit:
    try:
        dados_normalizados = scaler.transform(dados)
        if dados_normalizados.shape[1] != len(colunas_esperadas):
            st.error(f"Erro: Número de features ({dados_normalizados.shape[1]}) diferente do esperado ({len(colunas_esperadas)})")
        else:
            resultado = modelo.predict(dados_normalizados)[0]
            st.success("✅ Sem diabetes" if resultado == 0 else "⚠️ Risco de diabetes")
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")
        st.write("Detalhes técnicos:")
        st.write(f"Shape dos dados: {dados.shape}")
        st.write(f"Colunas: {dados.columns.tolist()}")
        st.write(f"Valores: {dados.values.tolist()}")
