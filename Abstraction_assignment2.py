#!/usr/bin/env python
# coding: utf-8

# In[3]:



# abstract base class work 
from abc import ABC, abstractmethod 


class A(ABC): 

    def B(self): 
        pass

class C(A): 

    def B(self): 
        print("Hello1") 

class D(A): 

    def B(self): 
        print("Hello2") 

class E(A): 

    def B(self): 
         print("Hello3") 

class F(A): 

    def B(self): 
        print("Hello4") 


R = C() 
R.B() 

K = F() 
K.B() 

R = E() 
R.B() 

K = F() 
K.B() 


# In[ ]:




