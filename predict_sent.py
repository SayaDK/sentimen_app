import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

vocab_size = 4000
max_length = 200
trunc_type='post'
oov_tok = "<OOV>"

word_index=[]
with open("word_index.txt",'r') as indeks:
    for word in indeks:
        word_index.append(word)
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(word_index)

sentiment_model = load_model(r'32_50_model.h5')
@st.cache
def predictor(string):
    coba_tes = []
    coba_tes.append(string)
    coba_seq = tokenizer.texts_to_sequences(coba_tes)
    coba_pad = pad_sequences(coba_seq, maxlen=max_length)
    pred = sentiment_model.predict(coba_pad)
    if (np.round(pred)==1):
        return("Positif")
    else:
        return ("Negatif")
 