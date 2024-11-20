import streamlit as st
import pickle
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    with open('C:/Users/bahian.j/OneDrive - Procter and Gamble/Documents/New folder (2)/optimised models in pkl packages/xgboost_fhyd.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit app layout
st.title('Machine Learning Model Prediction')

# Input fields for features
feature1 = st.number_input('Feature 1', min_value=0.0, max_value=100.0, value=0.0)
feature2 = st.number_input('Feature 2', min_value=0.0, max_value=100.0, value=0.0)
feature3 = st.number_input('Feature 3', min_value=0.0, max_value=100.0, value=0.0)

# Button to make a prediction
if st.button('Predict'):
    # Prepare the input for the model
    input_data = np.array([[feature1, feature2, feature3]])  # Adjust based on your model
    
    # Make a prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.success(f'The predicted value is: {prediction[0]}')
