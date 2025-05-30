import streamlit as st
import pandas as pd
import joblib

# 1. Carregar modelo e scaler
try:
    modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
    scaler = joblib.load("modelo/scaler.pkl")
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {str(e)}")
    st.stop()

# 2. Verificar features do scaler
if not hasattr(scaler, 'feature_names_in_'):
    st.error("O scaler não contém informações das features originais!")
    st.stop()

colunas_esperadas = scaler.feature_names_in_.tolist()
st.sidebar.write(f"🔍 O modelo espera {len(colunas_esperadas)} features:")
st.sidebar.write(colunas_esperadas)

# 3. Formulário de entrada
with st.form("form_predicao"):
    st.header("Dados do Paciente")
    
    # Dados básicos
    age = st.slider("Idade", 1, 120, 45)
    bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=28.5)
    glucose = st.number_input("Glicemia em jejum (mg/dL)", min_value=50, max_value=300, value=100)
    
    # Dados categóricos
    gender = st.selectbox("Sexo", ["Feminino", "Masculino"])
    hypertension = st.selectbox("Hipertensão", ["Não", "Sim"])
    
    submit = st.form_submit_button("Prever Diabetes")

if submit:
    # 4. Criar dicionário de entrada APENAS com as 24 features esperadas
    entrada = {
        "Age": age,
        "BMI": bmi,
        "Fasting_Blood_Glucose": glucose,
        "Sex_Male": 1 if gender == "Masculino" else 0,
        "Hypertension_Yes": 1 if hypertension == "Sim" else 0,
        
        # Preencher outras features com valores padrão
        **{col: 0 for col in colunas_esperadas if col not in ["Age", "BMI", "Fasting_Blood_Glucose", "Sex_Male", "Hypertension_Yes"]}
    }
    
    # 5. Garantir a ordem exata e número correto de features
    dados = pd.DataFrame([entrada])[colunas_esperadas]
    
    # 6. Verificação final
    st.write("### Dados enviados ao modelo:")
    st.write(dados)
    st.write(f"Shape dos dados: {dados.shape}")
    
    if dados.shape[1] != 24:
        st.error(f"ERRO: Número incorreto de features ({dados.shape[1]}). Deveria ser 24!")
        st.stop()
    
    # 7. Predição
    try:
        dados_normalizados = scaler.transform(dados)
        resultado = modelo.predict(dados_normalizados)[0]
        st.success("✅ Sem diabetes" if resultado == 0 else "⚠️ Risco de diabetes")
    except Exception as e:
        st.error(f"Falha na predição: {str(e)}")
