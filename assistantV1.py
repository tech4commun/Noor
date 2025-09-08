#basic prototype of nlp
'''"why spacy?"
Production-Ready: Built for real-world applications, not just research. It's fast and efficient.
High Accuracy: Provides superior linguistic analysis (part-of-speech tagging, dependency parsing).
Built-in Entity Recognition: Has powerful, pre-trained models to instantly identify names, locations, dates, and more.
Easy-to-Use API: Offers a consistent and simple interface for processing text, speeding up development.
Multi-Language Support: Provides a strong foundation for when we add local languages like Hindi.
ML Integration: Designed to train custom machine learning models for specific tasks like understanding crop names or intents.
import spacy'''
import spacy 
# Load the English language model
nlp = spacy.load("en_core_web_sm")

# This is a mock function to simulate checking a database for a price.
# In the real version, this will call an API.
def get_crop_price(crop_name, location_name):
    # This is fake data for testing.
    prices = {
        "potato": {"patna": 25, "delhi": 30},
        "tomato": {"patna": 40, "delhi": 55},
        "rice": {"patna": 20, "delhi": 25}
    }
    # Try to get the price. Use .lower() to make it case-insensitive.
    try:
        price = prices[crop_name.lower()][location_name.lower()]
        return f"The price of {crop_name} in {location_name} is {price} rupees per kilo."
    except KeyError:
        return f"Sorry, I could not find the price for {crop_name} in {location_name}."

# Main program logic
print("Hello! I am your agricultural assistant. How can I help you?")
# In the final app, this text will come from the Speech-to-Text engine.
user_text = input("Please type your question: ")

# Process the user's text with spaCy
doc = nlp(user_text)

# Extract information - Let's look for crops and locations.
# We assume the first proper noun is a location (e.g., Patna) and a common noun is a crop (e.g., potato).
crop = None
location = None

for token in doc:
    # Simple rule: if it's a common noun, it might be a crop.
    if token.pos_ == "NOUN" and not crop:
        crop = token.text
    # If it's a proper noun, it might be a location.
    if token.pos_ == "PROPN" and not location:
        location = token.text

print(f"I think you asked about the crop: {crop}")
print(f"I think you asked about the location: {location}")

# If we found both, try to get the price!
if crop and location:
    response = get_crop_price(crop, location)
    print("Assistant:", response)
else:
    print("Sorry, I didn't understand. Please mention both a crop and a location, like 'price of potato in Patna'.")
