#!/usr/bin/env python
# coding: utf-8

# In[15]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


pip install textblob


# In[ ]:





# In[12]:


data = pd.read_csv('Chat_Data.csv')
data


# In[16]:


# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit and transform the text data
vectorized_data = vectorizer.fit_transform(data['Question'])

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(vectorized_data)


# In[8]:


# Plot histogram of word counts in the questions column
plt.hist(data['Question'].str.split().apply(len), bins=20)
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Word Counts in Questions')
plt.show()


# In[9]:


# Concatenate all questions into a single string
all_questions = ' '.join(data['Question'])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(all_questions)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Questions')
plt.show()


# In[10]:


# Perform sentiment analysis on the answers column
data['Sentiment'] = data['Answer'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plot the sentiment distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x='Sentiment', bins=20, kde=True)
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis of Answers')
plt.show()


# In[19]:


# Plot cosine similarity heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=data['Question'], yticklabels=data['Question'])
plt.title('Cosine Similarity Heatmap')
plt.show()


# In[ ]:




