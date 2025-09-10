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
import csv
from collections import defaultdict
import math

class AgriculturalAssistantV5:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
        # Set slower speaking rate for better comprehension
        self.tts_engine.setProperty('rate', 150)
        
        self.conversation_context = {
            'last_crop': None,
            'last_location': None,
            'last_intent': None,
            'user_preferences': {},
            'conversation_history': []
        }
        self.supported_languages = ['english', 'hindi']
        self.current_language = 'english'
        
        # Initialize ML components
        self.intent_classifier = None
        self.vectorizer = None
        self.load_ml_models()
        
        # Load data from CSV files
        self.price_data = self.load_mandi_price_data('data/mandi_prices.csv')
        self.weather_data = self.load_csv_data('data/weather_data.csv')
        self.advice_data = self.load_csv_data('data/crop_advice.csv')
        self.localization_data = self.load_csv_data('data/localization.csv')
        self.crop_varieties = self.load_csv_data('data/crop_varieties.csv')
        self.market_info = self.load_csv_data('data/market_info.csv')
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Create default CSV files if they don't exist
        self.create_default_data_files()
        
        # Load conversation history if exists
        self.load_conversation_history()
        
        # Create mapping between common crop names and commodity names
        self.crop_name_mapping = {
            'wheat': 'Wheat',
            'rice': 'Rice',
            'tomato': 'Tomato',
            'potato': 'Potato',
            'paddy': 'Paddy(Dhan)(Common)',
            'tur': 'Arhar (Tur/Red Gram)(Whole)',
            'gram': 'Bengal Gram(Gram)(Whole)',
            'urd': 'Black Gram (Urd Beans)(Whole)',
            'millet': 'Foxtail Millet(Navane)',
            'jaggery': 'Gur(Jaggery)',
            'wood': 'Wood',
            'cucumber': 'Cucumbar(Kheera)',
            'chilli': 'Green Chilli',
            'lemon': 'Lemon',
            'pumpkin': 'Pumpkin',
            'maize': 'Maize',
            'sugarcane': 'Sugarcane'
        }
    
    def load_mandi_price_data(self, filepath):
        """Load mandi price data from CSV file with the given structure"""
        data = []
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                # Convert column names to standard format
                df.columns = df.columns.str.replace('_x0020_', ' ')
                df.columns = df.columns.str.replace('Arrival_Date', 'Arrival Date')
                
                # Convert to list of dictionaries
                data = df.to_dict('records')
            return data
        except Exception as e:
            print(f"Error loading mandi price data: {e}")
            return []
    
    def create_default_data_files(self):
        """Create default CSV files if they don't exist"""
        # Mandi price data (sample structure based on your data)
        if not os.path.exists('data/mandi_prices.csv'):
            mandi_data = [
                ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Arrival Date', 'Min Price', 'Max Price', 'Modal Price'],
                ['Andhra Pradesh', 'Chittor', 'Punganur', 'Tomato', 'Hybrid', 'FAQ', '10-09-2025', '1000', '1400', '1200'],
                ['Andhra Pradesh', 'Chittor', 'Vepanjari', 'Gur(Jaggery)', 'Achhu', 'FAQ', '10-09-2025', '2500', '3000', '2700'],
                ['Andhra Pradesh', 'Krishna', 'Divi', 'Paddy(Dhan)(Common)', 'B P T', 'FAQ', '10-09-2025', '2300', '2400', '2350'],
                ['Bihar', 'Bhojpur', 'Aarah', 'Potato', 'Jyoti', 'FAQ', '10-09-2025', '2000', '2200', '2100'],
                ['Chandigarh', 'Chandigarh', 'Chandigarh(Grain/Fruit)', 'Cucumbar(Kheera)', 'Other', 'FAQ', '10-09-2025', '2000', '4500', '3300'],
                ['Chandigarh', 'Chandigarh', 'Chandigarh(Grain/Fruit)', 'Green Chilli', 'Other', 'FAQ', '10-09-2025', '2000', '3500', '2800']
            ]
            self.save_csv_data('data/mandi_prices.csv', mandi_data)
        
        # Weather data
        if not os.path.exists('data/weather_data.csv'):
            weather_data = [
                ['location', 'condition', 'temperature', 'humidity', 'rainfall', 'forecast'],
                ['patna', 'Sunny', '32', '65', '0', 'Partly cloudy tomorrow'],
                ['delhi', 'Partly cloudy', '35', '60', '0', 'Light rain expected'],
                ['pune', 'Cloudy', '28', '75', '10', 'Heavy rain warning'],
                ['bangalore', 'Rainy', '26', '85', '25', 'Continuing rain'],
                ['chittor', 'Hot', '38', '45', '0', 'Clear skies'],
                ['krishna', 'Humid', '34', '80', '5', 'Possible showers']
            ]
            self.save_csv_data('data/weather_data.csv', weather_data)
        
        # Crop advice data
        if not os.path.exists('data/crop_advice.csv'):
            advice_data = [
                ['crop', 'season', 'soil_type', 'advice_english', 'advice_hindi'],
                ['wheat', 'winter', 'loamy', 'Ensure proper irrigation and use nitrogen-based fertilizers during growth.', 'वृद्धि के दौरान उचित सिंचाई सुनिश्चित करें और नाइट्रोजन आधारित उर्वरकों का उपयोग करें।'],
                ['rice', 'monsoon', 'clayey', 'Maintain water level at 2-3 inches and control weeds regularly.', 'पानी का स्तर 2-3 इंच पर बनाए रखें और नियमित रूप से खरपतवार नियंत्रित करें।'],
                ['tomato', 'summer', 'well-drained', 'Ensure proper drainage and rotate crops to prevent diseases.', 'रोगों को रोकने के लिए उचित जल निकासी सुनिश्चित करें और फसलों को घुमाएं।'],
                ['potato', 'winter', 'sandy loam', 'Plant in well-drained soil and maintain consistent moisture.', 'अच्छी तरह से सूखा मिट्टी में लगाएं और लगातार नमी बनाए रखें।']
            ]
            self.save_csv_data('data/crop_advice.csv', advice_data)
        
        # Localization data
        if not os.path.exists('data/localization.csv'):
            localization_data = [
                ['key', 'english', 'hindi'],
                ['greeting', 'Hello! How can I help you today?', 'नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?'],
                ['price_not_found', 'Sorry, I don\'t have price information for {crop} in {location}', 'क्षमा करें, मेरे पास {location} में {crop} की कीमत की जानकारी नहीं है'],
                ['weather_not_found', 'Weather information not available for {location}', '{location} के लिए मौसम की जानकारी उपलब्ध नहीं है'],
                ['advice_not_found', 'No specific advice available for {crop}', '{crop} के लिए कोई विशिष्ट सलाह उपलब्ध नहीं है'],
                ['missing_crop', 'Please specify a crop', 'कृपया एक फसल निर्दिष्ट करें'],
                ['missing_location', 'Please specify a location', 'कृपया एक स्थान निर्दिष्ट करें'],
                ['goodbye', 'Goodbye! Have a great day!', 'अलविदा! आपका दिन शुभ हो!'],
                ['error', 'Sorry, I encountered an error. Please try again.', 'क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।'],
                ['listening', 'Listening...', 'सुन रहा हूँ...'],
                ['not_understood', 'Sorry, I didn\'t understand that.', 'क्षमा करें, मैं समझा नहीं।'],
                ['service_unavailable', 'Sorry, speech service is unavailable.', 'क्षमा करें, भाषण सेवा उपलब्ध नहीं है।'],
                ['no_speech', 'No speech detected.', 'कोई भाषण नहीं मिला।'],
                ['help_prompt', 'I can help with crop prices, weather information, and agricultural advice. What do you need?', 'मैं फसल की कीमतों, मौसम की जानकारी और कृषि सलाह में मदद कर सकता हूं। आपको क्या चाहिए?']
            ]
            self.save_csv_data('data/localization.csv', localization_data)
        
        # Crop varieties data
        if not os.path.exists('data/crop_varieties.csv'):
            varieties_data = [
                ['crop', 'variety', 'characteristics', 'yield', 'duration'],
                ['wheat', 'HD 3086', 'High yield, disease resistant', '5.5 t/ha', '120 days'],
                ['rice', 'Pusa Basmati', 'Aromatic, long grain', '4.5 t/ha', '135 days'],
                ['tomato', 'Pusa Ruby', 'High yield, firm fruit', '60 t/ha', '90 days'],
                ['potato', 'Kufri Jyoti', 'High yield, disease resistant', '25 t/ha', '100 days']
            ]
            self.save_csv_data('data/crop_varieties.csv', varieties_data)
        
        # Market information data
        if not os.path.exists('data/market_info.csv'):
            market_data = [
                ['location', 'market_name', 'contact', 'business_hours'],
                ['patna', 'Patna Grain Market', '0612-XXXXXX', '6 AM to 8 PM'],
                ['delhi', 'Azadpur Mandi', '011-XXXXXX', '24 hours'],
                ['pune', 'Market Yard', '020-XXXXXX', '5 AM to 9 PM'],
                ['bangalore', 'K.R. Market', '080-XXXXXX', '6 AM to 10 PM'],
                ['chittor', 'Chittor Mandi', '08572-XXXXX', '5 AM to 7 PM'],
                ['krishna', 'Krishna Market', '0866-XXXXXX', '6 AM to 8 PM']
            ]
            self.save_csv_data('data/market_info.csv', market_data)
    
    def load_csv_data(self, filepath):
        """Load data from a CSV file into a dictionary"""
        data = {}
        try:
            if os.path.exists(filepath):
                with open(filepath, mode='r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if len(row) > 0:
                            # Use the first column value as key
                            key = list(row.values())[0].lower()
                            data[key] = row
            return data
        except Exception as e:
            print(f"Error loading CSV file {filepath}: {e}")
            return {}
    
    def save_csv_data(self, filepath, data):
        """Save data to a CSV file"""
        try:
            with open(filepath, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(data)
        except Exception as e:
            print(f"Error saving CSV file {filepath}: {e}")
    
    def load_conversation_history(self):
        """Load conversation history from file if exists"""
        try:
            if os.path.exists('data/conversation_history.csv'):
                with open('data/conversation_history.csv', mode='r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    self.conversation_context['conversation_history'] = list(reader)
        except Exception as e:
            print(f"Error loading conversation history: {e}")
    
    def save_conversation_history(self):
        """Save conversation history to file"""
        try:
            with open('data/conversation_history.csv', mode='w', newline='', encoding='utf-8') as file:
                if self.conversation_context['conversation_history']:
                    fieldnames = self.conversation_context['conversation_history'][0].keys()
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.conversation_context['conversation_history'])
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def get_localized_text(self, key, **kwargs):
        """Get localized text based on the current language"""
        if key in self.localization_data:
            text = self.localization_data[key].get(self.current_language, 
                                                  self.localization_data[key].get('english', ''))
            # Format the text with any provided keyword arguments
            if kwargs:
                try:
                    text = text.format(**kwargs)
                except KeyError:
                    pass
            return text
        return key  # Return the key itself if not found
    
    def load_ml_models(self):
        """Load pre-trained ML models for intent classification"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Check if models exist, otherwise use rule-based
            if os.path.exists('models/vectorizer.joblib') and os.path.exists('models/intent_classifier.joblib'):
                self.vectorizer = joblib.load('models/vectorizer.joblib')
                self.intent_classifier = joblib.load('models/intent_classifier.joblib')
                print("ML models loaded successfully")
            else:
                print("ML models not found. Using rule-based classification")
                self.intent_classifier = None
        except Exception as e:
            print(f"Error loading ML models: {e}")
            self.intent_classifier = None
    
    def classify_intent(self, text):
        """Classify intent using ML model with fallback to rule-based"""
        if self.intent_classifier is not None:
            try:
                # Transform text using the trained vectorizer
                text_vec = self.vectorizer.transform([text])
                
                # Predict intent
                prediction = self.intent_classifier.predict(text_vec)
                probability = self.intent_classifier.predict_proba(text_vec).max()
                
                # Fall back to rule-based if confidence is low
                if probability < 0.4:
                    return self.classify_intent_rule_based(text)
                
                return prediction[0]
            except Exception as e:
                print(f"ML classification error: {e}")
        
        # Fallback to rule-based classification
        return self.classify_intent_rule_based(text)
    
    def classify_intent_rule_based(self, text):
        """Rule-based intent classification"""
        text = text.lower()
        
        # Hindi keywords
        hindi_price_words = ['भाव', 'मूल्य', 'दर', 'कीमत', 'लागत', 'बाजार', 'मंडी']
        hindi_weather_words = ['मौसम', 'बारिश', 'तापमान', 'वर्षा', 'गर्मी', 'सर्दी', 'आर्द्रता']
        hindi_advice_words = ['रोग', 'कीट', 'समस्या', 'सलाह', 'उपाय', 'जानकारी', 'उपचार', 'बीमारी']
        hindi_greeting_words = ['नमस्ते', 'हैलो', 'हाय', 'कैसे', 'धन्यवाद', 'शुक्रिया']
        hindi_variety_words = ['किस्म', 'प्रजाति', 'वैरायटी', 'बीज']
        hindi_market_words = ['बाजार', 'मंडी', 'बिक्री', 'खरीद']
        
        # English keywords
        english_price_words = ['price', 'cost', 'rate', 'bhav', 'market', 'mandi']
        english_weather_words = ['weather', 'rain', 'temperature', 'forecast', 'humidity', 'hot', 'cold']
        english_advice_words = ['disease', 'pest', 'problem', 'advice', 'help', 'treatment', 'solution']
        english_greeting_words = ['hello', 'hi', 'hey', 'howdy', 'thanks', 'thank']
        english_variety_words = ['variety', 'type', 'seed', 'species']
        english_market_words = ['market', 'mandi', 'sell', 'buy']
        
        # Check for intent based on language
        if self.current_language == 'hindi':
            if any(word in text for word in hindi_price_words):
                return "get_price"
            elif any(word in text for word in hindi_weather_words):
                return "get_weather"
            elif any(word in text for word in hindi_advice_words):
                return "get_advice"
            elif any(word in text for word in hindi_greeting_words):
                return "greeting"
            elif any(word in text for word in hindi_variety_words):
                return "get_variety_info"
            elif any(word in text for word in hindi_market_words):
                return "get_market_info"
        else:
            if any(word in text for word in english_price_words):
                return "get_price"
            elif any(word in text for word in english_weather_words):
                return "get_weather"
            elif any(word in text for word in english_advice_words):
                return "get_advice"
            elif any(word in text for word in english_greeting_words):
                return "greeting"
            elif any(word in text for word in english_variety_words):
                return "get_variety_info"
            elif any(word in text for word in english_market_words):
                return "get_market_info"
        
        return "unknown"
    
    def detect_language(self, text):
        """Detect language based on character set"""
        # Check for Devanagari script characters
        if re.search(r'[\u0900-\u097F]', text):
            return 'hindi'
        
        # Check for common Hindi words in Roman script
        hindi_roman_words = ['ka', 'ki', 'ke', 'mein', 'batao', 'kya', 'hai', 'hain', 
                            'chahiye', 'bhav', 'mausam', 'salah', 'rog', 'keet', 'samasya',
                            'kichad', 'pani', 'baarish', 'garmi', 'sardi']
        
        words = text.lower().split()
        hindi_word_count = sum(1 for word in words if word in hindi_roman_words)
        
        # If more than 20% of words are Hindi words in Roman script, it's Hindi
        if hindi_word_count > len(words) * 0.2:
            return 'hindi'
        
        return 'english'
    
    def extract_entities(self, text):
        """Extract crop, location, variety, and other entities from text"""
        text = text.lower()
        
        # Crop mapping with variations
        crop_map = {
            'wheat': ['wheat', 'gehun', 'गेहूं', 'गेहूँ'],
            'rice': ['rice', 'chawal', 'chaval', 'चावल', 'dhan', 'धान', 'paddy', 'paddy(dhan)(common)'],
            'tomato': ['tomato', 'tamatar', 'टमाटर'],
            'potato': ['potato', 'aloo', 'aalu', 'आलू', 'आलु'],
            'tur': ['tur', 'arhar', 'red gram', 'अरहर', 'तूर'],
            'gram': ['gram', 'chana', 'bengal gram', 'चना', 'चना दाल'],
            'urd': ['urd', 'black gram', 'urad', 'उड़द', 'काला चना'],
            'millet': ['millet', 'navane', 'foxtail millet', 'बाजरा', 'मिलेट'],
            'jaggery': ['jaggery', 'gur', 'गुड़', 'जैगरी'],
            'cucumber': ['cucumber', 'kheera', 'cucumbar', 'खीरा', 'ककड़ी'],
            'chilli': ['chilli', 'mirchi', 'green chilli', 'मिर्च', 'हरी मिर्च'],
            'lemon': ['lemon', 'nimbu', 'नींबू', 'लेमन'],
            'pumpkin': ['pumpkin', 'kaddu', 'कद्दू', 'पम्पकिन'],
            'maize': ['maize', 'makka', 'मक्का', 'corn', 'मकई'],
            'sugarcane': ['sugarcane', 'ganna', 'गन्ना']
        }
        
        # Location mapping with variations (states and districts)
        location_map = {
            'andhra pradesh': ['andhra pradesh', 'andhra', 'आंध्र प्रदेश'],
            'chittor': ['chittor', 'chittoor', 'चित्तूर'],
            'krishna': ['krishna', 'कृष्णा'],
            'kurnool': ['kurnool', 'कुर्नूल'],
            'nellore': ['nellore', 'नेल्लोर'],
            'bihar': ['bihar', 'बिहार'],
            'bhojpur': ['bhojpur', 'bhojpur', 'भोजपुर'],
            'chandigarh': ['chandigarh', 'चंडीगढ़'],
            'chattisgarh': ['chattisgarh', 'chhattisgarh', 'छत्तीसगढ़'],
            'balodabazar': ['balodabazar', 'बालोदाबाजार'],
            'bilaspur': ['bilaspur', 'बिलासपुर'],
            'patna': ['patna', 'पटना'],
            'delhi': ['delhi', 'dilli', 'दिल्ली'],
            'pune': ['pune', 'पुणे'],
            'bangalore': ['bangalore', 'bengaluru', 'बैंगलोर', 'बेंगलुरु'],
            'mumbai': ['mumbai', 'bombay', 'मुंबई'],
            'kolkata': ['kolkata', 'calcutta', 'कोलकाता']
        }
        
        # Variety mapping
        variety_map = {
            'hybrid': ['hybrid', 'हाइब्रिड'],
            'achhu': ['achhu', 'आच्छू'],
            'bpt': ['b p t', 'bpt', 'बीपीटी'],
            'sona': ['sona', 'सोना'],
            'local': ['local', 'स्थानीय'],
            'jyoti': ['jyoti', 'ज्योति']
        }
        
        found_crop = None
        found_location = None
        found_variety = None
        
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
                
        # Check for variety mentions
        for variety, variations in variety_map.items():
            for variation in variations:
                if variation in text:
                    found_variety = variety
                    break
            if found_variety:
                break
                
        # If no location found, use context
        if not found_location and self.conversation_context.get('last_location'):
            found_location = self.conversation_context['last_location']
            
        # If no crop found, use context
        if not found_crop and self.conversation_context.get('last_crop'):
            found_crop = self.conversation_context['last_crop']
            
        return found_crop, found_location, found_variety
    
    def update_conversation_context(self, intent, crop, location, user_text, response):
        """Update conversation context based on current interaction"""
        if crop:
            self.conversation_context['last_crop'] = crop
            
        if location:
            self.conversation_context['last_location'] = location
            
        self.conversation_context['last_intent'] = intent
        
        # Add to conversation history
        self.conversation_context['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'user_text': user_text,
            'intent': intent,
            'response': response,
            'language': self.current_language
        })
        
        # Keep only last 100 conversations
        if len(self.conversation_context['conversation_history']) > 100:
            self.conversation_context['conversation_history'] = self.conversation_context['conversation_history'][-100:]
        
        # Save conversation history periodically
        if len(self.conversation_context['conversation_history']) % 10 == 0:
            self.save_conversation_history()
    
    def get_crop_price_from_mandi(self, crop, location=None, variety=None):
        """Get crop price from mandi data based on location and variety"""
        # Map common crop name to commodity name
        commodity_name = self.crop_name_mapping.get(crop.lower(), crop)
        
        # Filter data based on commodity
        matching_records = [record for record in self.price_data 
                           if record.get('Commodity', '').lower() == commodity_name.lower()]
        
        # Further filter by location if provided
        if location:
            location = location.lower()
            matching_records = [record for record in matching_records 
                               if (record.get('District', '').lower() == location or 
                                   record.get('Market', '').lower() == location or
                                   record.get('State', '').lower() == location)]
        
        # Further filter by variety if provided
        if variety:
            variety = variety.lower()
            matching_records = [record for record in matching_records 
                               if record.get('Variety', '').lower() == variety]
        
        if not matching_records:
            return None
        
        # For multiple matches, return the most recent or most relevant
        # Sort by date (most recent first)
        try:
            matching_records.sort(key=lambda x: datetime.strptime(x.get('Arrival Date', ''), '%d-%m-%Y'), reverse=True)
        except:
            pass
        
        return matching_records[0]
    
    def get_crop_price(self, crop, location, variety=None):
        """Get crop price from mandi data"""
        price_record = self.get_crop_price_from_mandi(crop, location, variety)
        
        if price_record:
            market = price_record.get('Market', 'Unknown market')
            variety = price_record.get('Variety', 'Unknown variety')
            min_price = price_record.get('Min Price', 'N/A')
            max_price = price_record.get('Max Price', 'N/A')
            modal_price = price_record.get('Modal Price', 'N/A')
            date = price_record.get('Arrival Date', 'Unknown date')
            
            if self.current_language == 'hindi':
                return (f"{market} में {crop} ({variety}) की कीमत: "
                       f"न्यूनतम ₹{min_price}, अधिकतम ₹{max_price}, "
                       f"मोडल ₹{modal_price} प्रति क्विंटल (तारीख: {date})")
            else:
                return (f"Price of {crop} ({variety}) in {market}: "
                       f"Min ₹{min_price}, Max ₹{max_price}, "
                       f"Modal ₹{modal_price} per quintal (date: {date})")
        
        # If not found in mandi data, try fallback to our CSV data
        crop_key = crop.lower()
        location_key = location.lower()
        
        # Look for the price in our CSV data
        for key, data in self.price_data.items():
            if data.get('crop', '').lower() == crop_key and data.get('location', '').lower() == location_key:
                price = data.get('price', 'N/A')
                unit = data.get('unit', '')
                date = data.get('date', '')
                
                if self.current_language == 'hindi':
                    return f"{location} में {crop} का मौजूदा मूल्य ₹{price} प्रति {unit} है (तारीख: {date})"
                else:
                    return f"The current price of {crop} in {location} is ₹{price} per {unit} (date: {date})"
        
        # If not found, return error message
        return self.get_localized_text('price_not_found', crop=crop, location=location)
    
    def get_weather_info(self, location):
        """Get weather information from CSV data"""
        location_key = location.lower()
        
        # Look for the weather in our CSV data
        if location_key in self.weather_data:
            data = self.weather_data[location_key]
            condition = data.get('condition', 'N/A')
            temperature = data.get('temperature', 'N/A')
            humidity = data.get('humidity', 'N/A')
            rainfall = data.get('rainfall', 'N/A')
            forecast = data.get('forecast', 'N/A')
            
            if self.current_language == 'hindi':
                return (f"{location} में मौसम: {condition}, तापमान: {temperature}°C, "
                       f"नमी: {humidity}%, वर्षा: {rainfall}mm. पूर्वानुमान: {forecast}")
            else:
                return (f"Weater in {location}: {condition}, Temperature: {temperature}°C, "
                       f"Humidity: {humidity}%, Rainfall: {rainfall}mm. Forecast: {forecast}")
        
        # If not found, return error message
        return self.get_localized_text('weather_not_found', location=location)
    
    def get_agriculture_advice(self, crop, season=None):
        """Get agricultural advice from CSV data"""
        crop_key = crop.lower()
        
        # Look for the advice in our CSV data
        for key, data in self.advice_data.items():
            if data.get('crop', '').lower() == crop_key:
                # If season is specified, try to get season-specific advice
                if season and data.get('season', '').lower() == season.lower():
                    advice_key = f'advice_{self.current_language}'
                    advice = data.get(advice_key, data.get('advice_english', 'No advice available'))
                    
                    if self.current_language == 'hindi':
                        return f"{season} मौसम में {crop} के लिए सलाह: {advice}"
                    else:
                        return f"Advice for {crop} in {season} season: {advice}"
                
                # If no season specified or no season-specific advice, return general advice
                advice_key = f'advice_{self.current_language}'
                advice = data.get(advice_key, data.get('advice_english', 'No advice available'))
                
                if self.current_language == 'hindi':
                    return f"{crop} के लिए सलाह: {advice}"
                else:
                    return f"Advice for {crop}: {advice}"
        
        # If not found, return error message
        return self.get_localized_text('advice_not_found', crop=crop)
    
    def get_crop_variety_info(self, crop):
        """Get information about crop varieties from CSV data"""
        crop_key = crop.lower()
        varieties = []
        
        # Find all varieties for the crop
        for key, data in self.crop_varieties.items():
            if data.get('crop', '').lower() == crop_key:
                varieties.append(data)
        
        if varieties:
            if self.current_language == 'hindi':
                response = f"{crop} की प्रमुख किस्में:\n"
                for variety in varieties:
                    response += f"- {variety.get('variety', 'N/A')}: {variety.get('characteristics', 'N/A')}, उपज: {variety.get('yield', 'N/A')}, अवधि: {variety.get('duration', 'N/A')}\n"
            else:
                response = f"Major varieties of {crop}:\n"
                for variety in varieties:
                    response += f"- {variety.get('variety', 'N/A')}: {variety.get('characteristics', 'N/A')}, Yield: {variety.get('yield', 'N/A')}, Duration: {variety.get('duration', 'N/A')}\n"
            
            return response
        else:
            if self.current_language == 'hindi':
                return f"क्षमा करें, मेरे पास {crop} की किस्मों की जानकारी नहीं है"
            else:
                return f"Sorry, I don't have information about varieties of {crop}"
    
    def get_market_info(self, location):
        """Get market information from CSV data"""
        location_key = location.lower()
        markets = []
        
        # Find all markets in the location
        for key, data in self.market_info.items():
            if data.get('location', '').lower() == location_key:
                markets.append(data)
        
        if markets:
            if self.current_language == 'hindi':
                response = f"{location} में प्रमुख कृषि बाजार:\n"
                for market in markets:
                    response += f"- {market.get('market_name', 'N/A')}: संपर्क: {market.get('contact', 'N/A')}, व्यापार के घंटे: {market.get('business_hours', 'N/A')}\n"
            else:
                response = f"Major agricultural markets in {location}:\n"
                for market in markets:
                    response += f"- {market.get('market_name', 'N/A')}: Contact: {market.get('contact', 'N/A')}, Business hours: {market.get('business_hours', 'N/A')}\n"
            
            return response
        else:
            if self.current_language == 'hindi':
                return f"क्षमा करें, मेरे पास {location} में बाजारों की जानकारी नहीं है"
            else:
                return f"Sorry, I don't have information about markets in {location}"
    
    def listen_to_speech(self):
        """Capture and convert speech to text"""
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Speak listening prompt
                listening_text = self.get_localized_text('listening')
                print(listening_text)
                
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            
            # Detect language
            self.current_language = self.detect_language(text)
            print(f"Detected language: {self.current_language}")
            
            return text
        
        except sr.UnknownValueError:
            return self.get_localized_text('not_understood')
        except sr.RequestError:
            return self.get_localized_text('service_unavailable')
        except sr.WaitTimeoutError:
            return self.get_localized_text('no_speech')
    
    def speak_response(self, response):
        """Convert text response to speech"""
        print(f"Assistant: {response}")
        self.tts_engine.say(response)
        self.tts_engine.runAndWait()
    
    def process_query(self, text):
        """Process user query and generate response"""
        if text.lower() in ['exit', 'quit', 'stop', 'बंद', 'रुको']:
            return "exit"
            
        # Classify intent
        intent = self.classify_intent(text)
        print(f"Intent: {intent}")
        
        # Extract entities
        crop, location, variety = self.extract_entities(text)
        
        # Generate response based on intent
        if intent == "get_price":
            if crop and location:
                response = self.get_crop_price(crop, location, variety)
            else:
                if not crop:
                    response = self.get_localized_text('missing_crop')
                else:
                    response = self.get_localized_text('missing_location')
                    
        elif intent == "get_weather":
            if location:
                response = self.get_weather_info(location)
            else:
                response = self.get_localized_text('missing_location')
                    
        elif intent == "get_advice":
            if crop:
                response = self.get_agriculture_advice(crop)
            else:
                response = self.get_localized_text('missing_crop')
        
        elif intent == "get_variety_info":
            if crop:
                response = self.get_crop_variety_info(crop)
            else:
                response = self.get_localized_text('missing_crop')
        
        elif intent == "get_market_info":
            if location:
                response = self.get_market_info(location)
            else:
                response = self.get_localized_text('missing_location')
                    
        elif intent == "greeting":
            response = self.get_localized_text('greeting')
                
        else:
            if self.current_language == 'hindi':
                response = "क्षमा करें, मैं केवल कीमतों, मौसम, कृषि सलाह, किस्मों और बाजार जानकारी के बारे में मदद कर सकता हूं"
            else:
                response = "I can help with prices, weather, agricultural advice, variety information, and market details. Please try again."
        
        # Update conversation context
        self.update_conversation_context(intent, crop, location, text, response)
        
        return response

    def run(self):
        """Main loop for the assistant"""
        print("Agricultural Voice Assistant v5 Started!")
        print("Now with enhanced NLP capabilities and more agricultural information!")
        print("Available crops: wheat, rice, tomato, potato, maize, sugarcane, tur, gram, urd, millet, jaggery, cucumber, chilli, lemon, pumpkin")
        print("Available locations: andhra pradesh, chittor, krishna, kurnool, bihar, bhojpur, chandigarh, chattisgarh, patna, delhi, pune, bangalore, mumbai, kolkata")
        print("Press Enter to speak or type your query...")
        
        # Initial greeting
        greeting = self.get_localized_text('greeting')
        print(f"Assistant: {greeting}")
        self.tts_engine.say(greeting)
        self.tts_engine.runAndWait()
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() in ['quit', 'exit', 'बंद']:
                    goodbye_text = self.get_localized_text('goodbye')
                    self.speak_response(goodbye_text)
                    break
                    
                # Use speech recognition if user pressed Enter without typing
                if user_input == "":
                    user_text = self.listen_to_speech()
                    if any(user_text.startswith(prefix) for prefix in [
                        "Sorry", "No speech", "क्षमा", "कोई भाषण"
                    ]):
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
                    goodbye_text = self.get_localized_text('goodbye')
                    self.speak_response(goodbye_text)
                    break
                    
                # Speak the response
                self.speak_response(response)
                
            except KeyboardInterrupt:
                goodbye_text = self.get_localized_text('goodbye')
                self.speak_response(goodbye_text)
                break
            except Exception as e:
                print(f"Error: {e}")
                error_msg = self.get_localized_text('error')
                self.speak_response(error_msg)
        
        # Save conversation history before exiting
        self.save_conversation_history()

if __name__ == "__main__":
    assistant = AgriculturalAssistantV5()
    assistant.run()