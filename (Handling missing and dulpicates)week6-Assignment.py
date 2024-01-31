#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from faker import Faker

# Set a random seed for reproducibility
np.random.seed(42)

# Function to generate a synthetic dataset with missing values and duplicates
def generate_dataset(num_rows=1000, missing_percentage=0.1):
    fake = Faker()

    data = {
        'ID': np.arange(1, num_rows + 1),
        'Name': [fake.name() for _ in range(num_rows)],
        'Age': [np.nan if np.random.rand() < missing_percentage else np.random.randint(18, 65) for _ in range(num_rows)],
        'Salary': [np.nan if np.random.rand() < missing_percentage else np.random.randint(30000, 100000) for _ in range(num_rows)],
    }

    df = pd.DataFrame(data)

    # Introduce duplicates
    df = pd.concat([df, df.sample(n=int(0.1*num_rows), replace=True)])

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

# Generate a dataset with missing values and duplicates
synthetic_dataset = generate_dataset()

# Display the first few rows of the dataset
print(synthetic_dataset.head())


# In[22]:


import seaborn as sns


# In[2]:


pip install faker


# In[4]:


df=synthetic_dataset


# In[5]:


df


# In[13]:


df.describe()


# In[16]:


df.dtypes


# In[18]:


df.duplicated().sum()


# In[10]:


df['Name'].isnull().sum()


# In[11]:


df['Age'].isnull().sum()


# In[12]:


df['Salary'].isnull().sum()


# In[ ]:


#first remove the duplicate rows


# In[21]:


df=df[~df.duplicated()]
df


# In[24]:


#Let fill the missing values
sns.lineplot(data=df,x='Age',y='Salary')


# In[29]:


#there is no linear relationbetween columns age and Salary let we them based on the values of  own columns alone using mean
df['Age']=df['Age'].fillna(df['Age'].mean())


# In[30]:


df['Age'].isnull().sum()


# In[31]:


df['Salary']=df['Salary'].fillna(df['Salary'].mean())


# In[ ]:




