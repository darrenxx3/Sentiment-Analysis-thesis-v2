"""
utils.py

This module contains utility functions for the sentiment analysis application.
"""

import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model():
    """Load the trained DistilBERT model and tokenizer"""
    model_path = "model/distilbert_1_model3epoch.pkl"
    tokenizer_path = "model/tokenizer_3epoch.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    model.eval()
    return model, tokenizer
