import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import nltk
import random
import sqlite3
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Customer Service ChatBot", layout="wide")

df = pd.read_csv('Product_Menu.csv', encoding='latin-1')

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






st.markdown(
        """
        <style>
        .stButton>button {
            color: green;
        }
        .css-1fcmnwj h1 {
            color: green;
        }
        </style>
        """,
    unsafe_allow_html=True
    )

st.sidebar.markdown("<span style='color: black;'>CUSTOMER SERVICE CHATBOT DASHBOARD</span>", unsafe_allow_html=True)
st.sidebar.markdown('<div style="width: 120px; height: 120px; border-radius: 60%; background-color: lightblue; display: flex; justify-content: center; align-items: center;"><span style="font-size: 50px;">🤖</span></div>', unsafe_allow_html=True)
st.markdown('<div style="position: fixed; bottom: 0; width: 100%; text-align: center;"><p><a href="https://www.zimyellowpage.com/mobile/detail.php?post_id=4450">Contact Details @ bot.serv 2024</a></p></div>', unsafe_allow_html=True)


dataset_expander = st.sidebar.expander("Our Menu")


# Display the product menu inside the sidebar expander
with dataset_expander:
    st.write("Product Menu/Location/Prices")
    st.write(df)
st.markdown("<span style='color: blue;'>PROMOTIONS!</span>", unsafe_allow_html=True)

promotion_product = random.choice(list(df['Product']))
promotion_us_price = df.loc[df['Product'] == promotion_product, '$US price'].values[0]
promotion_zig_price = df.loc[df['Product'] == promotion_product, 'ZiG price'].values[0]
        

st.info(f"Our  promotion for {promotion_product}, is still running. Prices has been reduced buy 5% . Its for all currency, either ZiG or $US. Visit the menu and buy Now!")
st.markdown("<span style='color: blue;'>Check also on the 'Updates'to see product recommendations!</span>", unsafe_allow_html=True)
st.subheader('Messaging Dialogue')
st.error('I am your AI bot, I may not be able to understand some of your questions. Just refer to the dashboard or our main site on the link provided for more information and products.You can train me on "Train.py"')

# Function to recommend products
def recommend_product(product_type):
    products = [data["Product"] for data in dataset if product_type.lower() in data["Product"].lower()]
    if products:
        return "We recommend the following products: " + ", ".join(products)
    else:
        return "Sorry, we currently don't offer products of that type."
    # Same as before
products = df['Product'].unique()
    
#st.sidebar.markdown("<span style='color: blue;'>Your Carrier Opportunity is here</span>", unsafe_allow_html=True)
#selected_preference = st.sidebar.selectbox("Product Catalogue", [""] + list(products))
st.sidebar.markdown('<a href="https://www.facebook.com/EatnLickZim/menu"><span style="color: blue;">Visit our site for more information</span></a>', unsafe_allow_html=True)
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

    if max_similarity < 0.4:
        return "I'm not sure about that. Could you please rephrase the question?"

    if 'Product' in response_data:
        return recommend_product(response_data["Product"])
    elif 'Location' in response_data:
        return find_nearest_location("User's location")
    else:
        return response_data["Answer"]


# Function to display the cosine similarity visualization
          # Randomly select a product for the promotion

# Create an empty list to store conversation history
# Create an empty list to store conversation history
# Create an empty list to store conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input('You:', key='user_input')
if st.button('Send') and user_input:
    response = bot_response(user_input)
    
    # Insert new dialogue pair just below user input
    st.write('You:', user_input)
    st.write('DelieBot:', response)
    
    # Append the user input and bot response to the conversation list
    st.session_state.conversation.insert(0, {'User': user_input, 'Bot': response})  # Insert at the beginning of the list

    # Display the conversation history
    for i, message in enumerate(st.session_state.conversation):
        if i == 0:
            st.markdown('<hr style="border: 2px solid blue;">', unsafe_allow_html=True)  # Add a colored line or separator for the current dialogue
        st.write('You:', message['User'])  # Display user input
        st.write('Bot:', message['Bot'])  # Display bot response
        st.write("--------------")  # Add a separator between dialogue pairs
#if 'conversation' not in st.session_state:
#    st.session_state.conversation = []

#user_input = st.text_input('You:', key='user_input')

# Up
     # Check if the bot asks for an answer or clarification
    if "I'm not sure about that. Could you please provide an answer?" in response:
        new_answer = st.text_input("Bot: I'm not sure about that. Could you please provide an answer?")
        if new_answer:
            question = user_input.lower()
            question = lemmatizer.lemmatize(question)
            update_dataset(question, new_answer)
            st.write("Bot: Thanks for the new information. I will remember that!")


# Function to display the recommended products
def display_recommended_products():
    with st.sidebar.expander("Updates"):
        st.subheader("Recommended Products")
        # Randomly select 3 products from the product menu
        recommended_products = random.sample(list(df['Product']), 3)
        us_prices = [df.loc[df['Product'] == p, '$US price'].values[0] for p in recommended_products]
        zig_prices = [df.loc[df['Product'] == p, 'ZiG price'].values[0] for p in recommended_products]

        for product, us_price, zig_price in zip(recommended_products, us_prices, zig_prices):
            st.write(f"- {product} - US Price: {us_price}, ZiG Price: {zig_price}")
        
                # Randomly select a product for the promotion
        promotion_product = random.choice(list(df['Product']))
        promotion_us_price = df.loc[df['Product'] == promotion_product, '$US price'].values[0]
        promotion_zig_price = df.loc[df['Product'] == promotion_product, 'ZiG price'].values[0]
        

        st.info(f"You are on the right spot! Checkout {promotion_product} for your special offer. Visit the menu and buy Now!")
display_recommended_products()
# Function to handle user feedback
def handle_user_feedback():
    with st.sidebar.expander("Feedback"):
        feedback_text = st.text_area("Leave your feedback for the chatbot:")
        if st.button("Submit Feedback"):
            # Connect to the SQLite database
            conn = sqlite3.connect('feedback.db')
            c = conn.cursor()

            # Create the feedback table if it doesn't exist
            c.execute('''CREATE TABLE IF NOT EXISTS feedback
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, feedback TEXT)''')

            # Insert the feedback into the database
            c.execute("INSERT INTO feedback (feedback) VALUES (?)", (feedback_text,))
            conn.commit()
            conn.close()
            st.success("Thank you for your feedback!")
handle_user_feedback()


