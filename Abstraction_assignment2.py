#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# abstract base class work 
from abc import ABC, abstractmethod 


class A(ABC): 

    def B(self): 
        pass

class C(anim): 

    def B(self): 
        print("Hello1") 

class D(anim): 

    def B(self): 
        print("Hello2") 

class E(anim): 

    def B(self): 
         print("Hello3") 

class F(anim): 

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

