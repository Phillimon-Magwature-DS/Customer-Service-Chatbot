import csv
import nltk
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset from CSV file
dataset = []
with open('Chat_Data.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        dataset.append(row)

# Initial setup
lemmatizer = WordNetLemmatizer()
corpus = []
responses = {}

# Preprocess the dataset
for data in dataset:
    corpus.append(data["Question"])
    responses[data["Question"]] = data

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Function to recommend products
def recommend_product(product_type):
    products = [data["Product"] for data in dataset if product_type.lower() in data["Product"].lower()]
    if products:
        return "We recommend the following products: " + ", ".join(products)
    else:
        return "Sorry, we currently don't offer products of that type."

# Function to find nearest location
def find_nearest_location(user_location):
    locations = [data["Location"] for data in dataset]
    # Implement logic to find the nearest location based on the user location
    # You can use geospatial calculations or any other method based on your dataset structure
    # For this example, we are just returning a random location from the dataset
    return random.choice(locations)

# Chatbot function
def bot_response(user_query):
    user_query = user_query.lower()
    user_query = lemmatizer.lemmatize(user_query)
    user_tfidf = vectorizer.transform([user_query])

    # Calculate similarity between user query and dataset questions
    similarities = cosine_similarity(user_tfidf, X)
    max_similarity = similarities.max()
    
    if max_similarity < 0.2:
        return "I'm sorry, I didn't understand that. Could you please rephrase the question?"

    # Get the most similar question from the dataset
    idx = similarities.argmax()
    response_data = list(responses.values())[idx]

    if 'Product' in response_data:
        product_type = response_data["Product"]
        return recommend_product(product_type)
    elif 'Location' in response_data:
        user_location = "User's location"  # You can replace this with user input or geolocation
        return find_nearest_location(user_location)
    else:
        return response_data["Answer"]

# Example of how the chatbot could work
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    else:
        reply = bot_response(user_input)
        print("Bot:", reply)
