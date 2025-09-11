# ml_model/intent_classifier.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

class IntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()
        self.classes = ['get_price', 'get_weather', 'get_advice', 'greeting', 'unknown']
        
    def train(self, training_data_path):
        # Load and preprocess training data
        df = pd.read_csv(training_data_path)
        X = self.vectorizer.fit_transform(df['text'])
        y = df['intent']
        
        # Train the model
        self.model.fit(X, y)
        
        # Save the model
        joblib.dump(self.vectorizer, 'models/vectorizer.joblib')
        joblib.dump(self.model, 'models/intent_classifier.joblib')
        
    def load_model(self):
        self.vectorizer = joblib.load('models/vectorizer.joblib')
        self.model = joblib.load('models/intent_classifier.joblib')
        
    def predict(self, text):
        # Transform text using the trained vectorizer
        text_vec = self.vectorizer.transform([text])
        
        # Predict intent
        prediction = self.model.predict(text_vec)
        probability = self.model.predict_proba(text_vec).max()
        
        # Fall back to rule-based if confidence is low
        if probability < 0.6:
            return self.rule_based_fallback(text)
        
        return prediction[0], probability
    
    def rule_based_fallback(self, text):
        # Fallback to the rule-based classifier from v3
        text = text.lower()
        if any(word in text for word in ['price', 'cost', 'rate', 'bhav']):
            return "get_price", 0.5
        elif any(word in text for word in ['weather', 'rain', 'temperature']):
            return "get_weather", 0.5
        elif any(word in text for word in ['disease', 'pest', 'problem', 'advice']):
            return "get_advice", 0.5
        else:
            return "unknown", 0.5