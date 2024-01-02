#!/usr/bin/env python
# coding: utf-8

# In[22]:


class Bank:
    pin1='abcd'
    def __init__(self,pin,withdrawam,depositam,availbalanceam):
        if pin==Bank.pin1:
                self.a=withdrawam
                self.c=depositam
                self.b=availbalanceam
        else:
            pass
        
            
        
    def withdrawal(self):
        print(self.a)
        self.b=self.b-self.a
        print(self.b)
    def deposit(self):
        print(self.c)
        self.b=self.b+self.c
        print(self.b)
    def balance(self):
        return(self.b)
A=Bank('abcd',500,1000,10000)
A.deposit()
A.withdrawal()
A.balance()

    
        
        
    


# In[ ]:




