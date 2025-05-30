import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Carregar modelo e scaler
try:
    modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
    scaler = joblib.load("modelo/scaler.pkl")
except Exception as e:
    st.error(f"Erro ao carregar modelo: {str(e)}")
    st.stop()

# 2. Obter features esperadas
if not hasattr(scaler, 'feature_names_in_'):
    st.error("Scaler não contém informações das features originais!")
    st.stop()

colunas_esperadas = scaler.feature_names_in_.tolist()
required_features = len(colunas_esperadas)
st.sidebar.write(f"🔢 O modelo requer {required_features} features:")
st.sidebar.write(colunas_esperadas)

# 3. Formulário de entrada
with st.form("diabetes_form"):
    st.header("Formulário de Avaliação")
    
    # Dados básicos (exemplos - adapte para suas 24 features)
    entrada = {
        'Age': st.slider("Idade", 20, 100, 45),
        'BMI': st.number_input("IMC", 15.0, 50.0, 25.0),
        'Fasting_Blood_Glucose': st.number_input("Glicemia (mg/dL)", 70, 300, 100),
        # Adicione TODAS as 24 features aqui conforme listado em colunas_esperadas
    }
    
    # Preencher automaticamente features não coletadas
    for col in colunas_esperadas:
        if col not in entrada:
            entrada[col] = 0  # Valor padrão para features não coletadas
    
    submit = st.form_submit_button("Realizar Predição")

# 4. Processamento
if submit:
    # Criar DataFrame com ordem correta
    dados = pd.DataFrame([entrada])[colunas_esperadas]
    
    # Verificação crítica
    st.write("### Dados enviados:")
    st.write(dados)
    st.write(f"Shape dos dados: {dados.shape}")
    
    if dados.shape[1] != required_features:
        missing = set(colunas_esperadas) - set(dados.columns)
        st.error(f"ERRO CRÍTICO: Faltam {len(missing)} features!")
        st.error(f"Features faltantes: {missing}")
        st.stop()
    
    # Predição
    try:
        dados_normalizados = scaler.transform(dados)
        if dados_normalizados.shape[1] != required_features:
            st.error(f"Erro na normalização: shape {dados_normalizados.shape}")
            st.stop()
            
        resultado = modelo.predict(dados_normalizados)[0]
        st.success("✅ Baixo risco" if resultado == 0 else "⚠️ Alto risco de diabetes")
    except Exception as e:
        st.error(f"Falha na predição: {str(e)}")
