import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Passos Mágicos | Risco Educacional",
    layout="centered"
)

# =========================================================
# CACHE — CARREGAMENTO DO MODELO
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("../model/rf_model.pkl")


rf_model = load_model()


# =========================================================
# CACHE — SHAP EXPLAINER (LEVE E ESTÁVEL)
# =========================================================
@st.cache_resource
def load_explainer(_model):
    background = pd.DataFrame({
        "ian": [4.0],
        "ida": [6.0],
        "ieg": [6.0],
        "ips": [5.0],
        "iaa": [5.0],
        "ipv": [6.0]
    })
    return shap.Explainer(_model, background)

explainer = load_explainer(rf_model)

# =========================================================
# TÍTULO E CONTEXTO
# =========================================================
st.title("📊 Predição de Risco Educacional")
st.markdown("""
Esta aplicação estima a **probabilidade de um aluno entrar em alto risco educacional**,
permitindo **intervenção pedagógica antecipada** pela equipe da Passos Mágicos.
""")

# =========================================================
# INPUTS DO USUÁRIO
# =========================================================
st.sidebar.header("📌 Indicadores do Aluno")

ian = st.sidebar.slider("IAN – Defasagem", 0.0, 10.0, 4.0)
ida = st.sidebar.slider("IDA – Desempenho Acadêmico", 0.0, 10.0, 6.0)
ieg = st.sidebar.slider("IEG – Engajamento", 0.0, 10.0, 6.0)
ips = st.sidebar.slider("IPS – Psicossocial", 0.0, 10.0, 5.0)
iaa = st.sidebar.slider("IAA – Autoavaliação", 0.0, 10.0, 5.0)
ipv = st.sidebar.slider("IPV – Ponto de Virada", 0.0, 10.0, 6.0)
ipp = st.sidebar.slider("IPP – Psicopedagogico", 0.0, 10.0, 6.0)

input_data = pd.DataFrame([{
    "ian": ian,
    "ida": ida,
    "ieg": ieg,
    "ips": ips,
    "iaa": iaa,
    "ipv": ipv,
    "ipp": ipp
}])

# =========================================================
# PREDIÇÃO
# =========================================================
prob_risco = rf_model.predict_proba(input_data.values)[0][1]

st.subheader("📈 Resultado da Análise")

st.metric(
    label="Probabilidade de Alto Risco",
    value=f"{prob_risco:.1%}"
)

# =========================================================
# REGRA DE NEGÓCIO (DECISÃO)
# =========================================================
if prob_risco >= 0.7:
    st.error("⚠️ Alto risco educacional — intervenção imediata recomendada")
elif prob_risco >= 0.4:
    st.warning("🟡 Risco moderado — acompanhamento pedagógico sugerido")
else:
    st.success("🟢 Baixo risco educacional")

# =========================================================
# EXPLICABILIDADE (SHAP — OPCIONAL)
# =========================================================
if st.checkbox("🔍 Mostrar explicação do modelo"):
    st.subheader("Fatores que mais influenciaram a predição")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_data.values)

    # ✅ Modelo single-output
    if isinstance(shap_values, list):
        # Para classificação binária, queremos a classe 1 (Alto Risco)
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Array (samples, features, classes) -> classe 1
        values = shap_values[:, :, 1]
    else:
        values = shap_values

        # Seleciona o primeiro sample
    if values.ndim > 1:
        values = values[0]

        # Garante 1D
    if values.ndim > 1:
        values = values.flatten()

    # Usar as colunas do DataFrame de input é mais seguro que `feature_names_in_`
    features = input_data.columns

    if len(values) != len(features):
        st.error(
            f"Inconsistência entre features ({len(features)}) "
            f"e valores SHAP ({len(values)})."
        )
        st.stop()

    shap_df = pd.DataFrame({
        "Feature": features,
        "Impacto SHAP": values
    }).sort_values("Impacto SHAP", key=abs, ascending=False)

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["Impacto SHAP"])
    ax.invert_yaxis()
    ax.set_xlabel("Impacto no risco de defasagem")
    st.pyplot(fig)

    st.caption(
        "Impactos positivos aumentam a probabilidade de risco; "
        "impactos negativos reduzem essa probabilidade."
    )

# =========================================================
# RODAPÉ
# =========================================================
st.markdown("---")
st.caption(
    "Modelo preditivo desenvolvido para o Datathon Passos Mágicos | "
    "Uso educacional e preventivo."
)
