import pickle
import numpy as np
import streamlit as st

st.title('Prediksi Kenaikan Permukaan Air Laut Terhadap Banjir')                         

TinggiLaut = st.text_input('Ketinggian Laut')      
KecepatanAngin = st.text_input('Kecepatan Angin')    
GangguanCuaca = st.text_input('Gangguan Cuaca')      
Astronomi = st.text_input('Astronomi')   

prediksibanjir = ''
if st.button('Prediksi Kenaikan Permukaan Laut Terhadap Banjir'):
    prediksibanjir = model.predict([[TinggiLaut, KecepatanAngin, GangguanCuaca, Astronomi]])
    if (prediksibanjir[0]==1):
        prediksibanjir = 'Terprediksi Kenaikan Permukaan Laut berpotensi Banjir'
    else:
         prediksibanjir = 'Terprediksi Kenaikan Permukaan Laut berpotensi Banjir'
st.success(prediksibanjir)    
