#!/usr/bin/env python
# coding: utf-8

# #Measures of central Tendency for a given data

# In[1]:


#Mean(It is the sum of all given numbers(data) divide by the total number observation)
import numpy as np
a=[1,2,3,4,5,6,76,53,7,4,5,6,6]
np.mean(a)


# In[2]:


#median(It is the  middlest term or number when the given numbers are arranged in ascending or decsending order)
import statistics
a=[1,2,3,4,5,6,76,53,7,4,5,6,6]
statistics.median(a)


# In[3]:


#Mode(It is the number which is repeated most times in the given data)
import statistics
a=[1,2,3,4,5,6,76,53,7,4,5,6,6]
statistics.mode(a)


# #Measures of Dispersion

# In[4]:


#Range(Diffrence of the highest and the lowest value in the data)
a=[1,2,3,4,5,6,76,53,7,4,5,6,6]
range=max(a)-min(a)
print(range)


# In[5]:


#Variance(It is the totalsum of difference of each number from the mean divide by total number of observations )
import statistics
a=[1,2,3,4,5,6,76,53,7,4,5,6,6]
statistics.variance(a)


# In[6]:


#standard deviation(square root of variance)
import statistics
a=[1,2,3,4,5,6,76,53,7,4,5,6,6]
np.std(a)


# In[ ]:




