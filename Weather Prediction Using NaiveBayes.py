#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# In[11]:


df=pd.read_csv("weather.csv")
df


# In[12]:


numerics=LabelEncoder()
df["outlook"]=numerics.fit_transform(df["outlook"])
df["temperature"]=numerics.fit_transform(df["temperature"])
df["humidity"]=numerics.fit_transform(df["humidity"])
df["windy"]=numerics.fit_transform(df["windy"])
target=df["play"]
df=df.drop("play",axis=1)
print(df)


# In[13]:


target


# In[15]:


classifier=GaussianNB()
classifier.fit(df,target)


# In[16]:


classifier.score(df,target)


# In[21]:


classifier.predict([[0,0,0,0]])


# In[ ]:




