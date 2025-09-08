'''How Intent Classification Works in This Program:
Keyword-Based Pattern Matching:
The classify_intent() function scans user input for specific keywords
Price queries: 'price', 'cost', 'rate', 'bhav'
Weather queries: 'weather', 'rain', 'temperature'
Advice queries: 'disease', 'pest', 'problem', 'advice'

Entity Extraction:
The extract_entities() function identifies crops and locations from the query
Currently uses simple string matching against predefined lists
Can be enhanced with more sophisticated NLP techniques

Query Routing:
Based on the classified intent, queries are routed to appropriate functions:
Price queries → get_crop_price()
Weather queries → get_weather_info()
Advice queries → get_agriculture_advice()

Response Generation:
Each specialized function processes the request
Returns appropriate responses based on available data
Responses are converted to speech using TTS'''

import speech_recognition as sr
import pyttsx3
from datetime import datetime

class AgriculturalAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
    def classify_intent(self, text):
        """Classify user intent from text input"""
        text = text.lower()
        if any(word in text for word in ['price', 'cost', 'rate', 'bhav']):
            return "get_price"
        elif any(word in text for word in ['weather', 'rain', 'temperature']):
            return "get_weather"
        elif any(word in text for word in ['disease', 'pest', 'problem', 'advice']):
            return "get_advice"
        else:
            return "unknown"

    def get_crop_price(self, crop, location):
        """Get crop price information (mock data)"""
        # Mock price database - replace with actual API
        price_data = {
            "wheat": {"patna": "₹2,100 per quintal", "delhi": "₹2,250 per quintal"},
            "rice": {"patna": "₹3,000 per quintal", "delhi": "₹3,200 per quintal"}
        }
        
        crop = crop.lower()
        location = location.lower()
        
        if crop in price_data and location in price_data[crop]:
            return f"The current price of {crop} in {location} is {price_data[crop][location]}"
        else:
            return f"Sorry, I don't have price information for {crop} in {location}"

    def get_weather_info(self, location):
        """Get weather information (mock data)"""
        # Mock weather data - replace with actual API
        weather_data = {
            "patna": "Sunny, 32°C",
            "delhi": "Partly cloudy, 35°C"
        }
        location = location.lower()
        return weather_data.get(location, "Weather information not available for this location")

    def get_agriculture_advice(self, crop):
        """Get agricultural advice (mock data)"""
        # Mock advice database - replace with actual API
        advice_data = {
            "tomato": "Ensure proper drainage and rotate crops to prevent diseases.",
            "potato": "Plant in well-drained soil and maintain consistent moisture."
        }
        crop = crop.lower()
        return advice_data.get(crop, "No specific advice available for this crop")

    def listen_to_speech(self):
        """Capture and convert speech to text"""
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5)
                
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
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

    def extract_entities(self, text):
        """Extract crop and location from text (basic implementation)"""
        text = text.lower()
        crops = ['wheat', 'rice', 'tomato', 'potato']
        locations = ['patna', 'delhi']
        
        found_crop = None
        found_location = None
        
        for crop in crops:
            if crop in text:
                found_crop = crop
                break
                
        for location in locations:
            if location in text:
                found_location = location
                break
                
        return found_crop, found_location

    def process_query(self, text):
        """Process user query and generate response"""
        if text.lower() in ['exit', 'quit', 'stop']:
            return "exit"
            
        intent = self.classify_intent(text)
        crop, location = self.extract_entities(text)
        
        if intent == "get_price":
            if crop and location:
                return self.get_crop_price(crop, location)
            else:
                return "Please specify both crop and location for price information."
        elif intent == "get_weather":
            if location:
                return self.get_weather_info(location)
            else:
                return "Please specify a location for weather information."
        elif intent == "get_advice":
            if crop:
                return self.get_agriculture_advice(crop)
            else:
                return "Please specify a crop for agricultural advice."
        else:
            return "I can help with crop prices, weather information, and agricultural advice. Please try again."

    def run(self):
        """Main loop for the assistant"""
        print("Agricultural Voice Assistant v3.0 Started!")
        print("Available crops: wheat, rice, tomato, potato")
        print("Available locations: patna, delhi")
        print("Press Enter to speak or type your query...")
        
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() == 'quit':
                    break
                    
                # Use speech recognition if user pressed Enter without typing
                if user_input == "":
                    user_text = self.listen_to_speech()
                    if user_text.startswith("Sorry"):
                        self.speak_response(user_text)
                        continue
                else:
                    user_text = user_input
                
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
                self.speak_response("Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    assistant = AgriculturalAssistant()
    assistant.run()