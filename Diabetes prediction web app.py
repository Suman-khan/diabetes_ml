import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('D:/[_ML PROJECT/Diabetes Prediction Deploy/trained_model.sav', 'rb'))

# Function for Prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)
    prediction = loaded_model.predict(input_data_as_numpy_array)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

def main():
    st.title('Diabetes Prediction Web App')

    # Getting user inputs
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Prediction result
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
