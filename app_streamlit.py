import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 1. Configuração da página
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺")
st.title("🩺 Preditor de Diabetes")

# 2. Carregar modelo e scaler
try:
    modelo = joblib.load("modelo/modelo_randomforest_diabetes.pkl")
    scaler = joblib.load("modelo/scaler.pkl")
except Exception as e:
    st.error(f"Erro ao carregar os arquivos do modelo: {str(e)}")
    st.stop()

# 3. Verificar estrutura do scaler
if not hasattr(scaler, 'feature_names_in_'):
    st.error("O scaler não contém informações das features originais. Recarregue o scaler corretamente.")
    st.stop()

colunas_esperadas = scaler.feature_names_in_.tolist()
st.sidebar.markdown(f"**Features esperadas ({len(colunas_esperadas)}):**")
st.sidebar.write(colunas_esperadas)

# 4. Função para criar dados de entrada
def criar_dados_entrada():
    st.header("Formulário de Entrada")
    
    entrada = {}
    
    # Dados numéricos
    if 'Age' in colunas_esperadas:
        entrada['Age'] = st.slider("Idade", 1, 120, 45)
    if 'BMI' in colunas_esperadas:
        entrada['BMI'] = st.number_input("IMC", min_value=10.0, max_value=60.0, value=28.5)
    
    # Dados clínicos
    if 'Fasting_Blood_Glucose' in colunas_esperadas:
        entrada['Fasting_Blood_Glucose'] = st.number_input("Glicemia em jejum (mg/dL)", min_value=50, max_value=300, value=100)
    if 'HbA1c' in colunas_esperadas:
        entrada['HbA1c'] = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.5)
    
    # Variáveis categóricas
    if 'Sex_Male' in colunas_esperadas:
        gender = st.selectbox("Sexo", ["Feminino", "Masculino"])
        entrada['Sex_Male'] = 1 if gender == "Masculino" else 0
    
    # Preencher outras features com zero
    for col in colunas_esperadas:
        if col not in entrada:
            entrada[col] = 0.0
    
    return entrada

# 5. Interface principal
dados_entrada = criar_dados_entrada()

if st.button("🔍 Prever Diabetes"):
    try:
        # Criar DataFrame com ordem exata
        dados = pd.DataFrame([dados_entrada])[colunas_esperadas]
        
        # Verificação final
        st.write("### Dados enviados ao modelo:")
        st.write(dados)
        
        if dados.shape[1] != len(colunas_esperadas):
            st.error(f"Erro: Número de features ({dados.shape[1]}) diferente do esperado ({len(colunas_esperadas)})")
            st.stop()
        
        # Normalização e predição
        dados_normalizados = scaler.transform(dados)
        resultado = modelo.predict(dados_normalizados)[0]
        
        st.success("✅ Resultado: Sem sinais de diabetes" if resultado == 0 else "⚠️ Resultado: Risco de diabetes detectado")
    
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")
        st.write("### Detalhes do erro:")
        st.write(f"Shape dos dados: {dados.shape if 'dados' in locals() else 'N/A'}")
        st.write(f"Colunas: {dados.columns.tolist() if 'dados' in locals() else 'N/A'}")
        st.write(f"Colunas esperadas: {colunas_esperadas}")
