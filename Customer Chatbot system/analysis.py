import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns

# Load the dataset
chat_data = pd.read_csv('Chat_Data.csv')

# Calculate cosine similarity matrix
# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit and transform the text data
vectorized_data = vectorizer.fit_transform(chat_data['Question'])

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(vectorized_data)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title("Data Exploration Options")
graph_options = st.sidebar.multiselect("Select Graphs to Display", ('Bar Chart', 'Word Cloud', 'Sentiment Analysis', 'Cosine Similarity'))

# Histogram
if 'Bar Chart' in graph_options:
    st.header("Bar Chart of Word Counts in Questions")
    plt.hist(chat_data['Question'].str.split().apply(len), bins=20)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Counts in Questions')
    st.pyplot()

# Word Cloud
if 'Word Cloud' in graph_options:
    st.header("Word Cloud of Questions")
    all_questions = ' '.join(chat_data['Question'])
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(all_questions)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Questions')
    st.pyplot()

# Sentiment Analysis
if 'Sentiment Analysis' in graph_options:
    st.header("Sentiment Analysis of Answers")
    chat_data['Sentiment'] = chat_data['Answer'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=chat_data, x='Sentiment', bins=20, kde=True)
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.title('Sentiment Analysis of Answers')
    st.pyplot()
# Cosine Similarity
#if 'Cosine Similarity' in graph_options:
#    st.header("Cosine Similarity Heatmap")
    # Plot cosine similarity heatmap
#    plt.figure(figsize=(8, 6))
#    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=chat_data['Question'], yticklabels=chat_data['Question'])
#    plt.title('Cosine Similarity Heatmap')
#    st.pyplot()
