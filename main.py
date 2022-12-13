import predict_sent as prediction
import streamlit as st

st.header('Prediksi Sentimen Ulasan Pengguna Aplikasi Okejek')

text = st.text_input('Masukkan Teks Ulasan', 'Makanan enak sekali')
if st.button('Mulai Analisis'):
    if text.strip()=='' :
        st.error('Cek kembali teks ulasan', icon="🚨")
    else:
        sentimen = prediction.predictor(text)
        st.write("sentimen "+sentimen)