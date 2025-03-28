import csv
#import streamlit as st
import nltk
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

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



vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)


# Function to recommend products
def recommend_product(product_type):
    products = [data["Product"] for data in dataset if product_type.lower() in data["Product"].lower()]
    if products:
        return "We recommend the following products: " + ", ".join(products)
    else:
        return "Sorry, we currently don't offer products of that type."
    # Same as before

# Function to find nearest location
def find_nearest_location(user_location):
    locations = [data["Location"] for data in dataset]
    # Implement logic to find the nearest location based on the user location
    # You can use geospatial calculations or any other method based on your dataset structure
    # For this example, we are just returning a random location from the dataset
    return random.choice(locations)



# Function to update the dataset with new questions and answers or update existing answers
def update_dataset(question, answer):
    global corpus, responses, vectorizer, X
    question_key = None
    for key, value in responses.items():
        if value["Question"] == question:
            question_key = key
            break
    if question_key:
        responses[question_key]["Answer"] = answer
    else:
                # Keep the existing fields intact when adding a new question
        new_data = {"Question": question, "Answer": answer}
        responses[question] = new_data
        corpus.append(question)
#        corpus.append(question)
#        responses[question] = {"Question": question, "Answer": answer}
    
        # Update the CSV file with the new dataset
    with open('Chat_Data.csv', 'w', newline='') as file:
        fieldnames = ['Question', 'Answer']
#        fieldnames = ['Question', 'Answer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
#        for data in responses.values():
#            data_to_write = {'Question': data['Question'], 'Answer': data['Answer']}
#            writer.writerow(data_to_write)

        for data in responses.values():
            writer.writerow(data)
    vectorizer = TfidfVectorizer(stop_words='english')
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
        # If the similarity is low, ask the user to provide an answer and update the dataset
        new_answer = input("Bot: I'm not sure about that. Could you please provide an answer? ")
        update_dataset(user_query, new_answer)
        return "Thanks for the new information. I will remember that!"

    # Get the index of the most similar question
    idx = similarities.argmax()
    response_data = list(responses.values())[idx]

    if max_similarity < 0.7:
        return "I'm not sure about that. Could you please rephrase the question?"

    if 'Product' in response_data:
        return recommend_product(response_data["Product"])
    elif 'Location' in response_data:
        return find_nearest_location("User's location")
    else:
        return response_data["Answer"]
  


while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    else:
        reply = bot_response(user_input)
        print("Bot:", reply)









