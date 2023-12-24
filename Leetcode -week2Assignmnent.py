#!/usr/bin/env python
# coding: utf-8

# In[89]:


#plus one
class Solution:
    def plusOne(self, digits: list[int]) -> list[int]:
        
               b=[str(i) for i in digits]
               c=int(''.join(b))+1
               c=str(c)
               c=[int(i) for i in c]
               return(c)
a=Solution()
a.plusOne([1,2,3])


# In[92]:


#28.Find the index of the first occurence in the string
class Solution:
    def strStr(self, a: str, b: str) -> int:
        c=0
        for i in range(0,len(a)-len(b)+1):
                    if a[i:i+len(b)]==b:
                            return(i)
                            c=c+1
        if c==0:
            return(-1)
a=Solution()
a.strStr('abgbjabc','abc')


# In[93]:


#242 Valid anogram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s)!=len(t):
            return(False)
        elif sorted(s)==sorted(t):
              return(True)
        else:    
            return(False)
a=Solution()
print(a.isAnagram('abcd','cdba'))
print(a.isAnagram('abcd','abba'))


# In[94]:


#Repeated substring pattern
class Solution:
    def rsp(self, s: str) -> bool:
    
                c=[]
                for i in range(1,(len(s)//2)+1):
                            if len(s)%i==0:
                                c.append(i)
                for i in c:
                    k=s[0:i]*(int(len(s)/i))
                    if k==s:
                        return(True)
                        break
                else:
                    return(False)
a=Solution()
print(a.rsp('ababab'))
print(a.rsp('aba'))


# In[95]:


#389.Find the difference
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
                        
                         for i in set(t):
                                if t.count(i)!=s.count(i):
                                        return(i)
a=Solution()
a.findTheDifference('asghe','ghesaa')


# In[96]:


#1822.Sign product of an array
class Solution:
    def arraySign(self, nums: list[int]) -> int:
        c=1
        for i in nums:
                c=c*i
        if c>0:
            return(1)
        elif c<0:
            return(-1)
        else:
            return(0)
a=Solution()
print(a.arraySign([1,2,3,4]))
print(a.arraySign([-1,-3,-4,5]))


# In[98]:


#1503.Can make arithmetic progression
class Solution:
    def cmap(self, a: list[int]) -> bool:
        a.sort()
        b=a[1]-a[0]
        for i in range(1,len(a)):
                if (a[i]-a[i-1])!=b:
                    return(False)
                    break
        else:
             return(True)
a=Solution()
print(a.cmap([1,2,3,4,5]))
print(a.cmap([1,2,5,8]))


# In[99]:


#896.Monotonic Array
class Solution:
    def isMonotonic(self, a: list[int]) -> bool:
        b=[]
        for i in range(1,len(a)):
            if a[i]>a[i-1]:
                b.append('>')
            elif a[i]==a[i-1]:
                b.append('.')
            else:
                b.append('<')
        if ('>'in b) and ('<'in b):
                 return(False)
        else:
            return(True)
a=Solution()
print(a.isMonotonic([1,1,1,2]))
print(a.isMonotonic([-2,-3,-2]))


# In[100]:


#13.Roman to integer
class Solution:
  def romanToInt(self, s: str) -> int:
    an = 0
    r = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
             'C': 100, 'D': 500, 'M': 1000}

    for a, b in zip(s, s[1:]):
      if r[a] < r[b]:
        an -= r[a]
      else:
        an += r[a]

    return an + r[s[-1]]
a=Solution()
a.romanToInt('IVCDM')


# In[103]:


#1768.Merge Strings alternatively
class Solution:
    def mergeAlternately(self, a: str, b: str) -> str:
        c=' '
        if len(a)>len(b):
            for i in range(len(b)):
                c=c+a[i]+b[i]
            c=c+a[len(b):]
        elif len(b)>len(a):
             for i in range(len(a)):
                c=c+a[i]+b[i]
             c=c+b[len(a):]
        else:
             for i in range(len(a)):
                c=c+a[i]+b[i]
               
        return(c[1:])
a=Solution()
a.mergeAlternately('abc','efgkkj')


# In[104]:


#283.Move Zeroes
class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
                        k = 0
                        for i in nums:
                             if i!= 0:
                                nums[k] = i
                                k += 1

                        for i in range(k, len(nums)):
                                     nums[i] = 0


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




