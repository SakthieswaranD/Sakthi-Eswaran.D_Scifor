#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
df = sns.load_dataset('tips')
df


# In[2]:


df.sample(10)


# In[3]:


df.head(5)


# In[4]:


df.tail(5)


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


x=df['total_bill']
y=df['tip']


# In[8]:


sns.barplot(x,y)


# In[9]:


sns.scatterplot(x,y)


# In[10]:


sns.histplot(x)


# In[11]:


plt.hist(x)
plt.xlabel('Hello')
plt.show()


# In[12]:


sns.boxplot(x)


# In[13]:


y = np.array([35, 25, 25, 15,14])
x=['a','b','c','d','e']
plt.pie(y,labels=x)
plt.show()

