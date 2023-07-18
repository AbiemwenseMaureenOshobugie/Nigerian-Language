import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_lottie import st_lottie
import requests

# Load the best_model and vectorizer from the pickle file
with open('best_model.pkl', 'rb') as f:
    loaded_objects = pickle.load(f)

loaded_model = loaded_objects['loaded_model']
vectorizer = loaded_objects['vectorizer']
label_encoder = loaded_objects['label_encoder']

# Create the Streamlit application
st.set_page_config(
    page_title="Nigeria Major Languages App",
    page_icon=":speech_balloon:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Apply styling and layout modifications
st.markdown(
    """
    <style>
    body {
        background-color: lightblue;
    }
    .header {
        background-color: skyblue;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .lottie-animation {
        width: 200px;
        height: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='header'><b>Nigeria Major Languages App</b></h1>", unsafe_allow_html=True)

# Display instructions
st.markdown(
    """
    Welcome to the Nigeria Major Languages App! This app is designed to detect the major languages spoken in Nigeria based on the input text.
    
    Instructions:
    1. Enter a text in <b>Yoruba</b>, <b>Igbo</b>, <b>Hausa</b>, or <b>Pidgin</b>.
    2. Click the 'Predict' button to see the predicted language.
    
    Please note that the accuracy of the prediction may vary depending on the complexity of the text and the quality of the language model used.
    """,
    unsafe_allow_html=True
)

# Create interactive components
text_input = st.text_input("Enter the text to classify:")
prediction_button = st.button("Predict")

# Perform prediction on user input
if prediction_button:
    input_text_reshaped = np.reshape([text_input], (-1, 1))

    # Get the text from input_text_reshaped
    input_text = input_text_reshaped[0][0]

    # Convert the input text to lowercase
    input_text_lower = input_text.lower()

    # Vectorize the input text using the loaded vectorizer
    input_text_vectorized = vectorizer.transform([input_text_lower])

    predicted_class = loaded_model.predict(input_text_vectorized)[0]
    predicted_language = label_encoder.inverse_transform([predicted_class])[0]

    # Display prediction with animation
    with st.spinner("Predicting..."):
        st.success("Predicted Language: " + predicted_language)
