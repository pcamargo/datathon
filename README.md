# 📊 FIAP - Datathon

Projeto de Machine Learning para prever risco educacional.

## 🛠️ Instalação

Certifique-se de ter o Python instalado e execute:
```bash
pip install -r requirements.txt
```

## 📓 Como executar os Notebooks

### 1. Execução Local
1.  Certifique-se de ter o Jupyter Notebook ou JupyterLab instalado (`pip install notebook`).
2.  Navegue até a pasta do projeto no seu terminal.
3.  Execute o comando:
    ```bash
    jupyter notebook
    ```
4.  Abra o arquivo `.ipynb` correspondente (ex: `notebooks/analise_modelagem.ipynb`).
5.  Execute as células sequencialmente para treinar o modelo.
    *   **Importante:** O notebook deve salvar o modelo treinado (`rf_model.pkl`) na pasta `model/` para que a aplicação funcione.

### 2. Google Colab
1.  Acesse Google Colab.
2.  Faça o upload do arquivo de notebook (`.ipynb`).
3.  Faça o upload da base de dados (`.csv`) para o armazenamento temporário do Colab.
4.  Execute as células.
5.  **Passo crucial:** Ao final da execução, faça o download do arquivo `rf_model.pkl` gerado pelo Colab e mova-o para a pasta `model/` no seu diretório local.

## 🚀 Como executar o Dashboard (Streamlit)

Para interagir com o modelo através da interface web:

1.  Abra o terminal na raiz do projeto.
2.  Execute o comando:
    ```bash
    streamlit run home.py
    ```
3.  O navegador abrirá automaticamente no endereço local (geralmente `http://localhost:8501`).

## ☁️ Como fazer o Deploy (Streamlit Cloud)

1.  **GitHub:** Suba seu código para um repositório público no GitHub.
    *   Garanta que o arquivo `requirements.txt` esteja na raiz com as dependências listadas.
    *   Garanta que o arquivo `model/rf_model.pkl` foi enviado (se for muito grande, use o Git LFS, mas para modelos simples o git comum aceita até 100MB).
2.  **Streamlit Cloud:**
    *   Acesse share.streamlit.io.
    *   Clique em **"New app"**.
    *   Selecione seu repositório, a branch e o arquivo principal (`home.py`).
    *   Clique em **"Deploy"**.
