'''Libraries and Their ML Functions

pandas
Loads structured training data from CSV files into DataFrames
Handles data preprocessing and manipulation for ML pipelines

TfidfVectorizer
Converts raw text into numerical feature vectors using TF-IDF scoring
Creates a document-term matrix where each word's importance is weighted by its frequency across documents
Handles text preprocessing tokenization internally

LogisticRegression
Implements a linear classification algorithm that estimates probability using a logistic function
Outputs probability distributions over multiple classes (multinomial classification)
Learns decision boundaries between different intent categories

train_test_split (imported but not used in this code)
Typically used to partition datasets into training and validation subsets
Helps evaluate model performance and prevent overfitting

joblib
Serializes trained ML models to disk for persistence
Enables model reuse without retraining (model deployment)
Maintains all learned parameters and vectorizer vocabulary mappings
'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Sample training data (in a real implementation, this would be a larger dataset)
training_data = [
    ("what is the price of wheat", "get_price"),
    ("how much does rice cost", "get_price"),
    ("tell me the weather in patna", "get_weather"),
    ("will it rain today", "get_weather"),
    ("my tomato plants have disease", "get_advice"),
    ("how to treat pests in wheat", "get_advice"),
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("good morning", "greeting"),
    ("what's the rate for potatoes", "get_price"),
    ("weather forecast for delhi", "get_weather"),
    ("need advice for rice cultivation", "get_advice"),
    ("नमस्ते", "greeting"),
    ("गेहूं का भाव", "get_price"),
    ("दिल्ली का मौसम", "get_weather"),
    ("टमाटर की बीमारी", "get_advice"),
    ("gehun ka bhav batao", "get_price"),
    ("tamatar ka bhav batao", "get_price"),
    ("delhi mein mausam kaisa hai", "get_weather"),
    ("patna ka temperature kya hai", "get_weather"),
    ("tamatar ki salah chahiye", "get_advice"),
    ("chawal ke rog ka upay", "get_advice"),
    ("aloo ke keet ka ilaj", "get_advice"),
    ("नमस्ते", "greeting"),
    ("हैलो", "greeting"),
    ("गेहूं का भाव", "get_price"),
    ("दिल्ली का मौसम", "get_weather"),
    ("टमाटर की सलाह", "get_advice")
]

# Create DataFrame
df = pd.DataFrame(training_data, columns=['text', 'intent'])

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text'])

# Train model
model = LogisticRegression()
model.fit(X, df['intent'])

# Save model and vectorizer
joblib.dump(vectorizer, 'models/vectorizer.joblib')
joblib.dump(model, 'models/intent_classifier.joblib')

print("Model trained and saved successfully")