import csv
import nltk
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarityhello

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
    responses[data["Question"]] = data["Answer"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

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
    response = list(responses.values())[idx]

    return response

# Example of how the chatbot could work
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    else:
        reply = bot_response(user_input)
        print("Bot:", reply)
