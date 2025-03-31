import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import string
import os

# GitHub URL for your model
model_url = "https://raw.githubusercontent.com/RahulBajaj7/Term-3-GroupProject/main/RNN/rb36tn52_sentiment_model.h5"
model_path = "rb36tn52_sentiment_model.h5"

# Download model if not present
if not os.path.exists(model_path):
    st.write("Downloading model from GitHub...")
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write("Model Downloaded Successfully.")
    except Exception as e:
        st.write(f"⚠️ Failed to download the model. Error: {e}")
        st.stop()

# Load the model
st.write("Loading the model...")
model = load_model(model_path)
st.write("Model Loaded Successfully!")

# Simple tokenizer function
def simple_tokenizer(text):
    # Lowercasing, removing punctuation, and splitting by spaces
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = text.split()
    return tokens

# Convert tokens to numbers (simple mapping)
word_index = {word: i + 1 for i, word in enumerate(set(" ".join(["good bad happy sad amazing terrible love hate excellent poor worst best satisfied disappointed"]).split()))}
vocab_size = len(word_index) + 1

def encode_text(text):
    tokens = simple_tokenizer(text)
    encoded_text = [word_index.get(word, 0) for word in tokens]
    return pad_sequences([encoded_text], maxlen=100)

# Function to predict sentiment
def predict_sentiment(review, threshold=0.7):
    # Encode and pad the input
    padded_seq = encode_text(review)
    
    # Predict sentiment
    prediction = model.predict(padded_seq)[0][0]
    sentiment = 'Positive' if prediction >= threshold else 'Negative'
    confidence = prediction
    
    # Display result
    st.write(f"**Review:** {review}")
    st.write(f"**Predicted Sentiment:** {sentiment}")
    st.write(f"**Confidence Score:** {confidence:.4f}")
    if confidence >= 0.9:
        st.write("Interpretation: Very strong confidence in the predicted sentiment.")
    elif confidence >= 0.7:
        st.write("Interpretation: Strong confidence in the predicted sentiment.")
    elif confidence >= 0.5:
        st.write("Interpretation: Moderate confidence in the predicted sentiment.")
    else:
        st.write("Interpretation: Low confidence in the predicted sentiment.")

# Streamlit UI
st.title("📊 Sentiment Analysis Using RNN")
st.write("Enter a product review below to predict its sentiment.")

# Input box for user review
user_review = st.text_area("Enter Review:", "")

# Predict button
if st.button("Predict Sentiment"):
    if user_review.strip():
        predict_sentiment(user_review)
    else:
        st.write("⚠️ Please enter a review to predict.")
