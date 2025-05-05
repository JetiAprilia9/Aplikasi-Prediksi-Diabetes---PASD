import pickle
import streamlit as st

# Membaca model dari file pickl
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

st.title("Prediksi Diabetes")

Glucose = st.number_input("Input Nilai Kadar Gula Darah: ",0.0)
BloodPressure = st.number_input("Input Nilai Tekanan Darah: ")
SkinThickness = st.number_input("Input Nilai Kelembapan Kulit: ")
Insulin = st.number_input("Input Nilai Insulin: ")
BMI = st.number_input("Input Nilai BMI: ")
DiabetesPedigreeFunction = st.number_input("Input Nilai Silsilah Diabetes: ")
Age = st.number_input("Input Nilai Umur: ",0)


DiabetesDiagnosis = ''

# Tombol Untuk Prediksi
if st.button("Prediksi"):
    DiabetesDiagnosis = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    
    if (DiabetesDiagnosis[0] == 1):
        DiabetesDiagnosis = 'Pasien Terdiagnosa Diabetes'
    else:
        DiabetesDiagnosis = 'Pasien Tidak Terdiagnosa Diabetes'
    
st.success(DiabetesDiagnosis)