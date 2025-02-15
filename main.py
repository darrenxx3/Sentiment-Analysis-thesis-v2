"""
main.py

This is the main entry point for the sentiment analysis application.
"""

import streamlit as st
import torch
import torch.nn.functional as F
from utils import load_model

# load the trained model
model, tokenizer = load_model()

# Streamlit UI
st.title("Sentiment Analysis Test")
st.subheader("Hey, use this sentiment analyzer!")

#user input
text = st.text_area("Type here","")

if st.button("Predict"):
    if text:    
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs =model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            sentiment = torch.argmax(probs, dim=-1).item()

            sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
            st.write("Results:")
            st.write(f"Sentiment: **{sentiment_labels[sentiment]}**")
    else:
        st.warning("Please enter text before predicting.😁")
