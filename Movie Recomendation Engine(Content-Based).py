#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[64]:


df = pd.read_csv("movie_dataset.csv")
df.head()


# In[65]:


features = ['keywords','cast','genres','director']


# In[66]:


for feature in features:
    df[feature] = df[feature].fillna('')


# In[67]:


def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']


# In[68]:


df["combined_fts"] = df.apply(combine_features,axis=1)
df.head()


# In[69]:


#create vector matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_fts'])


# In[70]:


#type(count_matrix)


# In[71]:


#count_matrix.toarray()


# In[72]:


#Compute cosine sim on count matrix
cosine_sim = cosine_similarity(count_matrix)


# In[73]:


#cv.get_feature_names()


# In[74]:


movie_user_likes = "Avatar"


# In[75]:


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# In[76]:


#Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)


# In[77]:


#Get a list of similar movies in descending order of similarity score
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_movie_list = sorted(similar_movies,key=lambda x:x[1],reverse=True)


# In[78]:


#Print titles of first 50 movies
count = 0
for index,sim in sorted_movie_list:
    count = count+1
    if count <= 50:
        print (get_title_from_index(index))


# In[ ]:




