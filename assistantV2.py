# assistant_v2.py
# Improved prototype using spaCy's Named Entity Recognition (NER)
# Processes text input (simulates speech) for crop price queries.
# Uses spaCy's NER to identify locations (GPE) and other entities.
# Falls back to noun extraction if NER doesn't find crops.
# Provides debug output of recognized entities.
# Queries mock database and generates spoken response.
# Example: "price of tomato in patna" → "The price of tomato in patna is 40 rupees/kilo"

import spacy
import re

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Mock function to get price
def get_crop_price(crop_name, location_name):
    prices = {
        "potato": {"patna": 25, "delhi": 30},
        "tomato": {"patna": 40, "delhi": 55},
        "rice": {"patna": 20, "delhi": 25}
    }
    try:
        price = prices[crop_name.lower()][location_name.lower()]
        return f"The price of {crop_name} in {location_name} is {price} rupees per kilo."
    except KeyError:
        return f"Sorry, I could not find the price for {crop_name} in {location_name}."

print("Hello! I am your agricultural assistant. How can I help you?")
user_text = input("Please type your question: ")

# Preprocess text: correct common misspellings and normalize case
user_text = user_text.lower()
user_text = re.sub(r'\bpice\b', 'price', user_text)  # Correct common misspelling

# Process the user's text with spaCy
doc = nlp(user_text)

# Initialize variables
crop = None
location = None

# Use Named Entity Recognition (NER)
print("\n--- Debug: Entities found by spaCy ---")
for ent in doc.ents:
    print(f"Text: {ent.text}, Label: {ent.label_} ({spacy.explain(ent.label_)})")
    # If the entity is a Geopolitical Entity (city, state, country), it's our location.
    if ent.label_ == "GPE":
        location = ent.text
    # For other entities, assume they might be the crop if we haven't found one yet
    elif not crop:
        crop = ent.text

# FALLBACK: Smarter crop identification if NER didn't find one
if not crop:
    # Create a list of potential crops from the document's nouns
    potential_crops = [token.text for token in doc if token.pos_ == "NOUN"]
    
    # Define a list of known crops to check against
    known_crops = ["tomato", "potato", "rice", "wheat", "corn", "onion"]
    
    # Look for any noun that is in our list of known crops
    for word in potential_crops:
        if word.lower() in known_crops:
            crop = word
            break
    
    # If no known crop is found, use the last noun instead of the first
    if not crop and potential_crops:
        crop = potential_crops[-1]

# FALLBACK: If NER didn't find a location, look for known locations in the text
if not location:
    known_locations = ["patna", "delhi", "mumbai", "kolkata", "chennai"]
    for token in doc:
        if token.text.lower() in known_locations:
            location = token.text
            break

# DEFAULT: If no location is provided, use a default based on our user persona
if not location:
    location = "Patna"  # Default to the main market for our user, Anil
    print(f"Using default location for your area: {location}")

print(f"\nI think you asked about the crop: {crop}")
print(f"I think you asked about the location: {location}")

if crop and location:
    response = get_crop_price(crop, location)
    print("\nAssistant:", response)
else:
    print("\nSorry, I didn't understand. Please mention both a crop and a location, like 'price of potato in Patna'.")