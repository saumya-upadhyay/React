import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load intents from JSON
with open("intents.json") as f:
    intents = json.load(f)

# Prepare patterns and tags
patterns = []
tags = []
tag_to_responses = {}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    tag_to_responses[intent["tag"]] = intent["responses"]

# Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

def get_response(user_input):
    # Vectorize user input
    user_vec = vectorizer.transform([user_input])
    
    # Calculate cosine similarity with known patterns
    similarities = cosine_similarity(user_vec, X)[0]
    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    # If similarity is too low, fallback
    if best_score < 0.3:
        return "I'm not sure I understand. Could you please rephrase?"

    # Get tag and return a random response
    tag = tags[best_match_index]
    return random.choice(tag_to_responses[tag])
