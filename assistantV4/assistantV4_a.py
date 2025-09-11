'''speech_recognition (sr)
Used for capturing audio from the microphone and converting speech to text.
sr.Recognizer() initializes the speech recognizer.
recognizer.listen(source) captures audio from the microphone.
recognizer.recognize_google(audio) uses Google's speech recognition API to convert audio to text.

pyttsx3
Text-to-speech conversion library.
pyttsx3.init() initializes the TTS engine.
tts_engine.say(response) queues the response to be spoken.
tts_engine.runAndWait() plays the queued speech.

pandas (pd) and numpy (np)
Not directly used in the provided code, but typically used for data manipulation and numerical operations.
Might be used if the ML models were trained using these libraries (but in this code, they are only loaded from disk).

sklearn.feature_extraction.text.TfidfVectorizer
Used for converting text into TF-IDF features.
In this code, it is loaded from a saved model (vectorizer.joblib) to transform the input text for the ML model.

sklearn.linear_model.LogisticRegression
A machine learning model for classification.
In this code, it is loaded from a saved model (intent_classifier.joblib) to predict the intent of the user's query.

sklearn.model_selection.train_test_split
Not directly used in the provided code, but typically used for splitting data into training and testing sets during model training.

joblib
Used for saving and loading Python objects (like the trained ML models).
joblib.load is used to load the pre-trained vectorizer and classifier.

requests
Not directly used in the provided code, but typically used for making HTTP requests to APIs (like weather, price, and translation APIs). The code has placeholders for API endpoints but uses mock implementations.

json
Not directly used in the provided code, but typically used for parsing JSON responses from APIs.

datetime.datetime
Used to get the current timestamp for updating the conversation context.

re
Regular expressions, used in detect_language to check for Devanagari script characters.

os
Not directly used in the provided code, but typically used for interacting with the operating system, e.g., file paths.'''

import speech_recognition as sr
import pyttsx3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import requests
import json
from datetime import datetime
import re
import os

class AgriculturalAssistantEnhanced:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.conversation_context = {
            'last_crop': None,
            'last_location': None,
            'last_intent': None
        }
        self.supported_languages = ['english', 'hindi']
        self.current_language = 'english'
        
        # Initialize ML components
        self.intent_classifier = None
        self.vectorizer = None
        self.load_ml_models()
        
        # API endpoints (replace with actual endpoints)
        self.weather_api_url = "https://api.weatherapi.com/v1/current.json"
        self.price_api_url = "https://api.agmarknet.gov.in/api/price"
        self.translation_api_url = "https://api.translate.com/v1/translate"
        
    def load_ml_models(self):
        """Load pre-trained ML models for intent classification"""
        try:
            self.vectorizer = joblib.load('models/vectorizer.joblib')
            self.intent_classifier = joblib.load('models/intent_classifier.joblib')
            print("ML models loaded successfully")
        except:
            print("ML models not found. Using rule-based classification as fallback")
            self.intent_classifier = None
    
    def classify_intent_ml(self, text):
        """Classify intent using ML model"""
        if self.intent_classifier is None:
            intent = self.classify_intent_rule_based(text)
            return intent, 0.5 if intent != "unknown" else 0.1
        
        # Transform text using the trained vectorizer
        text_vec = self.vectorizer.transform([text])
        
        # Predict intent
        prediction = self.intent_classifier.predict(text_vec)
        probability = self.intent_classifier.predict_proba(text_vec).max()
        
        # Fall back to rule-based if confidence is low
        if probability < 0.4:
            intent = self.classify_intent_rule_based(text)
            return intent, probability
        
        return prediction[0], probability
    
    def classify_intent_rule_based(self, text):
        """Rule-based intent classification (fallback)"""
        text = text.lower()
        
        # Hindi keywords
        if any(word in text for word in ['भाव', 'मूल्य', 'दर', 'कीमत', 'लागत']):
            return "get_price"
        elif any(word in text for word in ['मौसम', 'बारिश', 'तापमान', 'वर्षा']):
            return "get_weather"
        elif any(word in text for word in ['रोग', 'कीट', 'समस्या', 'सलाह', 'उपाय', 'जानकारी']):
            return "get_advice"
        elif any(word in text for word in ['नमस्ते', 'हैलो', 'हाय', 'कैसे']):
            return "greeting"
        
        # English keywords
        if any(word in text for word in ['price', 'cost', 'rate', 'bhav']):
            return "get_price"
        elif any(word in text for word in ['weather', 'rain', 'temperature', 'forecast']):
            return "get_weather"
        elif any(word in text for word in ['disease', 'pest', 'problem', 'advice', 'help']):
            return "get_advice"
        elif any(word in text for word in ['hello', 'hi', 'hey', 'howdy']):
            return "greeting"
        else:
            return "unknown"
    
    def detect_language(self, text):
        """Improved language detection based on character set and common words"""
        # Check for Devanagari script characters
        if re.search(r'[\u0900-\u097F]', text):
            return 'hindi'
        
        # Check for common Hindi words in Roman script
        hindi_roman_words = ['ka', 'ki', 'ke', 'mein', 'batao', 'kya', 'hai', 'hain', 
                            'chahiye', 'bhav', 'mausam', 'salah', 'rog', 'keet', 'samasya']
        
        words = text.lower().split()
        hindi_word_count = sum(1 for word in words if word in hindi_roman_words)
        
        # If more than 20% of words are Hindi words in Roman script, it's Hindi
        if hindi_word_count > len(words) * 0.2:
            return 'hindi'
        else:
            return 'english'
    
    def extract_entities(self, text):
        """Enhanced entity extraction with better pattern matching"""
        text = text.lower()
        
        # Crop mapping with variations
        crop_map = {
            'wheat': ['wheat', 'gehun', 'गेहूं', 'गेहूँ'],
            'rice': ['rice', 'chawal', 'chaval', 'चावल'],
            'tomato': ['tomato', 'tamatar', 'टमाटर'],
            'potato': ['potato', 'aloo', 'aalu', 'आलू', 'आलु']
        }
        
        # Location mapping with variations
        location_map = {
            'patna': ['patna', 'पटना'],
            'delhi': ['delhi', 'dilli', 'दिल्ली'],
            'pune': ['pune', 'पुणे'],
            'bangalore': ['bangalore', 'bengaluru', 'बैंगलोर', 'बेंगलुरु']
        }
        
        found_crop = None
        found_location = None
        
        # Check for crop mentions with all variations
        for crop, variations in crop_map.items():
            for variation in variations:
                if variation in text:
                    found_crop = crop
                    break
            if found_crop:
                break
                
        # Check for location mentions with all variations
        for location, variations in location_map.items():
            for variation in variations:
                if variation in text:
                    found_location = location
                    break
            if found_location:
                break
                
        # If no location found, use context
        if not found_location and self.conversation_context.get('last_location'):
            found_location = self.conversation_context['last_location']
            
        # If no crop found, use context
        if not found_crop and self.conversation_context.get('last_crop'):
            found_crop = self.conversation_context['last_crop']
            
        return found_crop, found_location
    
    def update_conversation_context(self, intent, crop, location):
        """Update conversation context based on current interaction"""
        if crop:
            self.conversation_context['last_crop'] = crop
            
        if location:
            self.conversation_context['last_location'] = location
            
        self.conversation_context['last_intent'] = intent
        self.conversation_context['timestamp'] = datetime.now().isoformat()
    
    def get_crop_price(self, crop, location):
        """Get crop price from API (mock implementation)"""
        price_data = {
            "wheat": {"patna": "₹2,100 per quintal", "delhi": "₹2,250 per quintal"},
            "rice": {"patna": "₹3,000 per quintal", "delhi": "₹3,200 per quintal"},
            "tomato": {"patna": "₹1,500 per quintal", "delhi": "₹1,800 per quintal"},
            "potato": {"patna": "₹1,200 per quintal", "delhi": "₹1,400 per quintal"}
        }
        
        crop = crop.lower()
        location = location.lower()
        
        if crop in price_data and location in price_data[crop]:
            price = price_data[crop][location]
            if self.current_language == 'hindi':
                return f"{location} में {crop} का मौजूदा मूल्य {price} है"
            else:
                return f"The current price of {crop} in {location} is {price}"
        else:
            if self.current_language == 'hindi':
                return f"क्षमा करें, मेरे पास {location} में {crop} की कीमत की जानकारी नहीं है"
            else:
                return f"Sorry, I don't have price information for {crop} in {location}"
    
    def get_weather_info(self, location):
        """Get weather information from API (mock implementation)"""
        weather_data = {
            "patna": "Sunny, 32°C",
            "delhi": "Partly cloudy, 35°C",
            "pune": "Cloudy, 28°C",
            "bangalore": "Rainy, 26°C"
        }
        
        location = location.lower()
        weather = weather_data.get(location, "Weather information not available for this location")
        
        if self.current_language == 'hindi':
            return f"{location} में मौसम: {weather}"
        else:
            return f"Weather in {location}: {weather}"
    
    def get_agriculture_advice(self, crop):
        """Get agricultural advice (mock implementation)"""
        advice_data = {
            "wheat": "Ensure proper irrigation and use nitrogen-based fertilizers during growth.",
            "rice": "Maintain water level at 2-3 inches and control weeds regularly.",
            "tomato": "Ensure proper drainage and rotate crops to prevent diseases.",
            "potato": "Plant in well-drained soil and maintain consistent moisture."
        }
        
        crop = crop.lower()
        advice = advice_data.get(crop, "No specific advice available for this crop")
        
        if self.current_language == 'hindi':
            return f"{crop} के लिए सलाह: {advice}"
        else:
            return f"Advice for {crop}: {advice}"
    
    def listen_to_speech(self):
        """Capture and convert speech to text"""
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5)
                
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            
            # Detect language
            self.current_language = self.detect_language(text)
            print(f"Detected language: {self.current_language}")
            
            return text
        except sr.UnknownValueError:
            return "Sorry, I didn't understand that."
        except sr.RequestError:
            return "Sorry, speech service is unavailable."
        except sr.WaitTimeoutError:
            return "No speech detected."
    
    def speak_response(self, response):
        """Convert text response to speech"""
        print(f"Assistant: {response}")
        self.tts_engine.say(response)
        self.tts_engine.runAndWait()
    
    def process_query(self, text):
        """Process user query and generate response"""
        if text.lower() in ['exit', 'quit', 'stop', 'बंद', 'रुको']:
            return "exit"
            
        # Classify intent using ML with fallback to rule-based
        intent, confidence = self.classify_intent_ml(text)
        print(f"Intent: {intent}, Confidence: {confidence:.2f}")
        
        # Extract entities
        crop, location = self.extract_entities(text)
        
        # Update conversation context
        self.update_conversation_context(intent, crop, location)
        
        # Generate response based on intent
        if intent == "get_price":
            if crop and location:
                return self.get_crop_price(crop, location)
            else:
                missing = []
                if not crop:
                    missing.append("crop")
                if not location:
                    missing.append("location")
                
                if self.current_language == 'hindi':
                    return f"कृपया {' और '.join(missing)} निर्दिष्ट करें"
                else:
                    return f"Please specify {' and '.join(missing)}"
                    
        elif intent == "get_weather":
            if location:
                return self.get_weather_info(location)
            else:
                if self.current_language == 'hindi':
                    return "कृपया मौसम जानकारी के लिए स्थान निर्दिष्ट करें"
                else:
                    return "Please specify a location for weather information"
                    
        elif intent == "get_advice":
            if crop:
                return self.get_agriculture_advice(crop)
            else:
                if self.current_language == 'hindi':
                    return "कृपया कृषि सलाह के लिए फसल निर्दिष्ट करें"
                else:
                    return "Please specify a crop for agricultural advice"
                    
        elif intent == "greeting":
            if self.current_language == 'hindi':
                return "नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?"
            else:
                return "Hello! How can I help you today?"
                
        else:
            if self.current_language == 'hindi':
                return "क्षमा करें, मैं केवल कीमतों, मौसम और कृषि सलाह के बारे में मदद कर सकता हूं"
            else:
                return "I can help with prices, weather, and agricultural advice. Please try again."

    def run(self):
        """Main loop for the assistant"""
        print("Agricultural Voice Assistant Enhanced Version Started!")
        print("Now with improved language detection and entity extraction!")
        print("Available crops: wheat, rice, tomato, potato")
        print("Available locations: patna, delhi, pune, bangalore")
        print("Press Enter to speak or type your query...")
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() in ['quit', 'exit', 'बंद']:
                    break
                    
                # Use speech recognition if user pressed Enter without typing
                if user_input == "":
                    user_text = self.listen_to_speech()
                    if user_text.startswith("Sorry") or user_text.startswith("No speech"):
                        self.speak_response(user_text)
                        continue
                else:
                    user_text = user_input
                    # Detect language from text input
                    self.current_language = self.detect_language(user_text)
                    print(f"Detected language: {self.current_language}")
                
                # Process the query
                response = self.process_query(user_text)
                
                if response == "exit":
                    break
                    
                # Speak the response
                self.speak_response(response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                error_msg = "Sorry, I encountered an error. Please try again."
                if self.current_language == 'hindi':
                    error_msg = "क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।"
                self.speak_response(error_msg)

if __name__ == "__main__":
    assistant = AgriculturalAssistantEnhanced()
    assistant.run()