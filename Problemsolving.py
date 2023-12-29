#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().set_next_input('What is the difference between list and tuple in Python');get_ipython().run_line_magic('pinfo', 'Python')

Explain the concept of PEP 8.

get_ipython().set_next_input('What is the purpose of the __init__ method in Python classes');get_ipython().run_line_magic('pinfo', 'classes')

How does inheritance work in Python? Provide an example.

Explain the difference between staticmethod and classmethod.

What is Polymorphism in Python? Give an example.

get_ipython().set_next_input('How do you handle exceptions in Python');get_ipython().run_line_magic('pinfo', 'Python')

Explain the Global Interpreter Lock (GIL) in Python.

What is a decorator in Python? Provide an example.
get_ipython().set_next_input('How do you implement encapsulation in Python');get_ipython().run_line_magic('pinfo', 'Python')

Explain the concept of duck typing.

get_ipython().set_next_input('What is the difference between append() and extend() methods for lists');get_ipython().run_line_magic('pinfo', 'lists')

get_ipython().set_next_input('How does the with statement work in Python');get_ipython().run_line_magic('pinfo', 'Python')

Discuss the use of self in Python classes.

Explain the purpose of the __slots__ attribute.

get_ipython().set_next_input('What is the difference between an instance variable and a class variable');get_ipython().run_line_magic('pinfo', 'variable')
get_ipython().set_next_input('How do you implement Encapsulation, Abstraction, Polymorphism');get_ipython().run_line_magic('pinfo', 'Polymorphism')
How do you Implement single level Inheritance, multiple level inheritance, multi level inheritance, Hybrid Inheritance


# 1.#What is the difference between list and tuple in Python?
# List is a collection of elements of same or different datatypes.List is mutable.example for list is [1,2,34].
# whereas tuple is also a collection of elements of same or different datatype,but tuple is immutable.example for tuple is (1,2,3).
# 

# 2.What is the purpose of the __init__ method in Python classes?
# __init__ is a constructor which is used is used to intialize all variables in the class

# In[ ]:


#3.How does inheritance work in Python? Provide an example.
#Inheritance is nothing but the  accessing the class by another class or in other terms it gives the properties(methods ,variables) of one class to another.


# #4.Explain the difference between staticmethod and classmethod.
# class methods are methods in the the arguments is cls keyword.these methods are directly called by using class name.
# static methods are methods in which the arguments may be self or cls with any number of argumnets.These methods can be called by class or objects.
# 

# In[3]:


#5.What is Polymorphism in Python? Give an example.
#polymorphism is that the methods in a class will return different output(or) having different properties with different arguments.
class animal:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def add(self):
        return (self.a+self.b)
a=animal(1,3)  
print(a.add())
b=animal('abc','def')
print(b.add())


# In[5]:


#6.What is the difference between append() and extend() methods for lists?
#append will add the value to the list at last position whereas as extend will add the values of other list to another list
a=[1,2,3]
b=[5,5,6]
a.append(b)
print(a)
a=[1,2,3]
a.extend(b)
print(a)


# In[ ]:


#7.How do you implement encapsulation in Python?
#Encapsulation is the method of wrapping up of variables and methods into single entity .Class itself a encapsulation
#example
class animal:
    a=100
    def hello(self):
        return ('hello')
#In the above we can see that we had encapsulated the variable and method in it.we can also encapsulated by making the members as public,private or protected which helps which objects should have the access to use this class.

        


# In[ ]:


#8.Discuss the use of self in Python classes.
#self is used as arguments for the instance methods in the classes.self in the methods arguments implies the class itself.


# In[ ]:


#9.What is the difference between an instance variable and a class variable?
#A variable is called as class variable which is defined outside of any methods in a class  and the variable which are defined inside any class method is instance variable.


# #10How does the with statement work in Python?
# with statements is used for file handling
# since  creating a varibale for file by,
# file=open('filename')
# we can directly use with method by
# with open('filename') as file:
#     

# #11.explain the purpose of slots in python.
# slots is a use to reudce the memory of the objects,It is a static type method in this no dynamic dictionary is required for allocating attribute.
# 
# 

# In[ ]:


#12.How do you implement Encapsulation, Abstraction, Polymorphism?
#Encapsulation implementation
class animal:
    a=100
    def hello(self):
        return ('hello')


# In[ ]:


#Polymorphism implementation
class animal:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def add(self):
        return (self.a+self.b)
a=animal(1,3)  #this give sum of two numbers
print(a.add())
b=animal('abc','def')  #This gives concatenation of two strings
print(b.add())


# In[6]:


#Abstraction implementation

class animal:
    a=100
    def hello(self):
        pass
class dog(animal):
    def hello(self):
        return("Hello")
A=dog()
A.hello()


# In[ ]:


#13.How do you Implement single level Inheritance, multiple level inheritance, multi level inheritance, Hybrid Inheritance


# In[7]:


#single level Inheritance implementation

class animal:
    
    def hello(self):
        return('hello')
class dog(animal):
    def Iam(self):
        return("I am dog")
A=dog()
A.hello()


# In[9]:


#Multi level Inheritance implementation
class human:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def hello(self):
        return(f'hello, I am {self.a}, I have {self.b} bike')
class tall(human):
    def talk(self):
        return("I am a tall person")
class bike(tall):
    def sound(self):
          return('pompom')
    
A=bike('mike','pulsar')

print(A.hello())
print(A.talk())
print(A.sound())


# In[10]:


#multiple level inheritance and hybrid inheritance
class human:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def hello(self):
        return(f'hello, I am {self.a}, I have {self.b} bike')
class tall(human):
    def talk(self):
        return("I am a tall person")
class fat(human):
    def obese(self):
        return('I am very fat')
class bike(tall,fat):
    def sound(self):
          return('pompom')
    
A=bike('mike','pulsar')

print(A.hello())
print(A.talk())
print(A.obese())
print(A.sound())


# In[ ]:




