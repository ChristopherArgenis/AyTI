# app.py

import streamlit as st
import pandas as pd
from bow_processor import Bow_Words

def main():
    st.title("Aplicación de Análisis de Texto (BoW)")
    st.write("Sube un archivo .txt para realizar análisis de texto con Bag of Words y TF-IDF.")

    uploaded_file = st.file_uploader("Elige un archivo .txt", type=["txt"])

    if uploaded_file is not None:
        st.success("Archivo cargado con éxito. Procesando...")

        # Instanciar la clase con el archivo cargado
        processor = Bow_Words(uploaded_file)

        # 1. Filtrar Stopwords
        st.subheader("Paso 1: Filtrado de Stopwords")
        filtered_words_list = processor.filter_stopwords()
        st.write("Texto sin stopwords (primeras 5 líneas):")
        st.code('\n'.join([' '.join(line) for line in filtered_words_list[:5]]))
        st.write("---")

        # 2. Lematización
        st.subheader("Paso 2: Lematización")
        lemmatized_text = processor.lemmatized(filtered_words_list)
        st.write("Texto lematizado (primeras 5 líneas):")
        st.code('\n'.join(lemmatized_text[:5]))
        st.write("---")

        # 3. Opción de selección para TF-IDF o N-Gram
        st.subheader("Paso 3: Vectorización")
        analysis_type = st.selectbox(
            "Selecciona el tipo de análisis:",
            ("TF-IDF", "N-Gram")
        )

        if analysis_type == "TF-IDF":
            st.info("Calculando TF-IDF...")
            feature_names, vectors = processor.tf_idf(lemmatized_text)
            df_result = processor.get_df(feature_names, vectors)
            
            st.write("### Matriz TF-IDF")
            st.dataframe(df_result)
        
        elif analysis_type == "N-Gram":
            ngram_size = st.slider("Selecciona el tamaño del N-Gram:", 1, 2, 3)
            st.info(f"Calculando N-Grams de tamaño {ngram_size}...")
            feature_names, vectors = processor.n_gram(lemmatized_text, ngram_size)
            df_result = processor.get_df(feature_names, vectors)
            
            st.write("### Matriz de Conteo (N-Gram)")
            st.dataframe(df_result)

        # 4. Suma de Frecuencias
        st.subheader("Paso 4: Suma de Frecuencias")
        if 'df_result' in locals():
            sum_series = processor.get_sum(df_result)
            df_sum = pd.DataFrame(sum_series, columns=['Frecuencia']).sort_values(by='Frecuencia', ascending=False)
            
            st.write("### Frecuencia Total de Palabras")
            st.dataframe(df_sum)
            
            st.download_button(
                label="Descargar DataFrame de Frecuencias como CSV",
                data=df_sum.to_csv().encode('utf-8'),
                file_name='frecuencias_palabras.csv',
                mime='text/csv'
            )