#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Single Inheritence
class Vehicle():
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def explain(self):
        print(f'I am {self.a} and I have {self.b}')
class B(Vehicle):
    def hello(self):
        print('Hello')
c=B('car','four wheeler')
c.explain()
c.hello()


# In[8]:


#Multi Inheritence
class Vehicle():
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def explain(self):
        print(f'I am {self.a} and I have {self.b}')
class D(Vehicle):
    def hello(self):
        print('Hello')
class B(D):
    def ask(self):
        print('who are you')
c=B('car','four wheeler')
c.explain()
c.hello()
c.ask()


# In[ ]:




