#!/usr/bin/env python
# coding: utf-8

# 40. Create a class called "Bank" with properties for "name" and "accounts". Add methods to
# create new
# accounts, deposit and withdraw money from accounts, and to calculate the total balance of all
# accounts.
# Create an object of this class and test out the input

# In[1]:


class Bank:
    def __init__(self,name,bal=0):
        self.name=name
        self.bal=bal
    def withdraw(self,a):
        self.a=a
        self.bal=self.bal-self.a
        return(a)
    def deposit(self,b):
        self.b=b
        self.bal=self.bal+self.b
        return(self.b)
    def balance(self):
        return(self.bal)
    
        
        


# In[2]:


A=Bank('abc')
print(A.deposit(700))
print(A.withdraw(200))
print(A.deposit(300))


# In[3]:


print(A.balance())


# In[ ]:




