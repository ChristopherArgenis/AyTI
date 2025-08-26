import streamlit as st

st.title("Aplicación de Procesamiento de Archivos de Texto")
st.write("Sube un archivo .txt para procesarlo y ver los resultados en un DataFrame.")

uploaded_file = st.file_uploader("Elige un archivo .txt", type=["txt"])

# Entrada con exito
if uploaded_file is not None:
        st.write("Archivo subido con éxito.")
