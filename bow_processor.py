import streamlit as st
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Descargar los recursos de NLTK solo una vez
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Cargar el modelo de spacy una sola vez
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Descargando el modelo de Spacy 'es_core_news_sm'...")
    from spacy.cli import download
    download("es_core_news_sm")
    nlps = spacy.load("es_core_news_sm")

class Bow_Words:
  nlp = nlps

  def __init__(self, text_file):
    # Lee el contenido del archivo de texto
    string_data = text_file.getvalue().decode("utf-8")
        
    # Divide el texto por líneas
    self.ds = string_data.splitlines()

  def filter_stopwords(self):
    spanish_stopwords = set(stopwords.words('spanish'))

    # Separar las palabras
    sep = []
    for d in self.ds:
      sep.append(d.split())

    # Filtrar las palabras
    filtered_sep = []
    for s in sep:
      filtered_words = [word for word in s if word.lower() not in spanish_stopwords]
      filtered_sep.append(filtered_words)

    return filtered_sep

  def lemmatized(self, filtered_sep):
    lemmatized_sep = []
    for s in filtered_sep:
      # Une la lista de palabras en un solo string
      doc = self.nlp(" ".join(s))
      lemmatized_words = [token.lemma_ for token in doc]
      lemmatized_sep.append(" ".join(lemmatized_words))

    return lemmatized_sep

  def tf_idf(self, lemmatized_sep):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(lemmatized_sep)
    return vectorizer.get_feature_names_out(), X.toarray()

  def n_gram(self, lemmatized_sep, select: int):
    match select:
      case 1:
        vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word')
      case 2:
        vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='word')
      case 3:
        vectorizer = CountVectorizer(ngram_range=(3, 3), analyzer='word')
      case _:
        vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word')
    X = vectorizer.fit_transform(lemmatized_sep)
    return vectorizer.get_feature_names_out(), X.toarray()

  def get_df(self, arrays_name:str, arrays_values):
    return pd.DataFrame(arrays_values, columns=arrays_name)

  def get_sum(self, df):
    return df.sum(axis=0)