import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.keras')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Convert words to integers
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Pad sequences to the same length
    return padded_review


# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    if user_input.strip() == '':
        st.write('Please enter a review text.')
    else:
        # Preprocess the user input and predict sentiment
        preprocessed_input = preprocess_text(user_input)

        # Make the prediction
        prediction = model.predict(preprocessed_input)

        # Interpret the result
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        sentiment_score = prediction[0][0]  # The model returns a score between 0 and 1

        # Display the result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {sentiment_score:.4f}')

else:
    st.write('Please enter a movie review.')
