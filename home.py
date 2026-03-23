import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(page_title="Teste de Modelo", layout="centered")

st.title("Teste de Modelo - Risco Educacional")

# =========================================================
# CARREGAMENTO DO MODELO
# =========================================================
@st.cache_resource
def load_model():
    try:
        # Carrega o mesmo modelo utilizado em home.py
        return joblib.load("model/rf_model.pkl")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

model = load_model()


# =========================================================
# CARREGAMENTO DO EXPLAINER (SHAP)
# =========================================================
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

if model:
    explainer = load_explainer(model)

# =========================================================
# SIDEBAR - INPUTS E CONTROLE
# =========================================================
st.sidebar.header("📌 Parâmetros do Aluno")

# Inputs baseados nas features do modelo (ian, ida, ieg, ips, iaa, ipv, ipp)
ian = st.sidebar.slider("IAN – Defasagem", 0.0, 10.0, 5.0)
ida = st.sidebar.slider("IDA – Desempenho Acadêmico", 0.0, 10.0, 5.0)
ieg = st.sidebar.slider("IEG – Engajamento", 0.0, 10.0, 5.0)
ips = st.sidebar.slider("IPS – Psicossocial", 0.0, 10.0, 5.0)
iaa = st.sidebar.slider("IAA – Autoavaliação", 0.0, 10.0, 5.0)
ipv = st.sidebar.slider("IPV – Ponto de Virada", 0.0, 10.0, 5.0)
ipp = st.sidebar.slider("IPP – Psicopedagógico", 0.0, 10.0, 5.0)

st.sidebar.markdown("---")
# Checkbox para controlar a exibição do resultado
mostrar_analise = st.sidebar.checkbox("Mostrar Resultado da Análise", value=True)

# =========================================================
# LÓGICA DE PREDIÇÃO
# =========================================================
if model and mostrar_analise:
    input_data = pd.DataFrame([{
        "ian": ian, "ida": ida, "ieg": ieg, "ips": ips,
        "iaa": iaa, "ipv": ipv, "ipp": ipp
    }])

    # Realiza a predição, convertendo para numpy array para evitar warning de feature names
    prob_risco = model.predict_proba(input_data.values)[0][1]

    st.subheader("📊 Análise em Tempo Real")
    st.metric(label="Probabilidade de Alto Risco", value=f"{prob_risco:.1%}")

    if prob_risco >= 0.07:
        st.error("⚠️ Alto risco educacional")
    elif prob_risco >= 0.04:
        st.warning("🟡 Risco moderado")
    else:
        st.success("🟢 Baixo risco educacional")

    # =========================================================
    # EXPLICABILIDADE (SHAP)
    # =========================================================
    # st.markdown("---")
    # st.subheader("🔍 Por que este resultado?")
    
    # Gera a explicação para os dados atuais
    # explanation = explainer(input_data)
    # shap_values = explanation[0, :, 1] # Classe 1 (Risco)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # shap.plots.waterfall(shap_values, show=False)
    # st.pyplot(fig)
    