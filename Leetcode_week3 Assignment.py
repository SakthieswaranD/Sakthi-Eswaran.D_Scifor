#!/usr/bin/env python
# coding: utf-8

# In[1]:


#58.length of last word in the string
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        a=s.split()
        return(len(a[-1]))
A=Solution()
print(A.lengthOfLastWord('asc sf 12334 asdefgg'))


# In[2]:


#709.To Lower  Case
class Solution:
    def toLowerCase(self, s: str) -> str:
        return(s.lower())
A=Solution()
print(A.toLowerCase('heLLOOO'))


# In[3]:


#682.Baseball game
import functools 
class Solution:
    def calPoints(self, a: list[str]) -> int:
        b=[]
        import functools
        for i in a:
            if i.isnumeric() :
                 b.append(int(i))
            elif list(i)[0]=='-':
                 b.append(int(i))
            elif i=='+':
                 b.append(b[-1]+b[-2])
            elif i=='C':
                 b.remove(b[-1])
            elif i=='D':
                 b.append(b[-1]*2)
            else:
                pass
        c=0
        for i in b:
            c=c+i
        return(c)
A=Solution()
print(A.calPoints(["1","C"]))
    


# In[4]:


a=6
b=13
c=0
for i in range(a,b+1):
    if i%2!=0:
         c=c+1
c


# In[5]:


#1491.Average salary excluding minimum and maximum salary
class Solution:
    def average(self, salary: list[int]) -> float:
        salary.remove(min(salary))
        salary.remove(max(salary))
        a=((sum(salary))/len(salary))
        return(a)
A=Solution()
print(A.average([100,2000,1000,2500]))


# In[6]:


#976.Largest perimeter triangle
class Solution:
  def largestPerimeter(self, nums: list[int]) -> int:
    nums = sorted(nums)

    for i in range(len(nums) - 1, 1, -1):
      if nums[i - 2] + nums[i - 1] > nums[i]:
        return nums[i - 2] + nums[i - 1] + nums[i]

    return 0
A=Solution()
print(A.largestPerimeter([2,1,2]))


# In[7]:


#1232.check if it is a straight line
class Solution:
  def checkStraightLine(self, coordinates: list[list[int]]) -> bool:
    x0, y0, x1, y1 = *coordinates[0], *coordinates[1]
    dx = x1 - x0
    dy = y1 - y0

    return all((x - x0) * dy == (y - y0) * dx for x, y in coordinates)
A=Solution()
print(A.checkStraightLine([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]))
    


# In[8]:


#67.Add binary
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        add= bin(int(a, 2) + int(b, 2))
        return((add[2:]))  
A=Solution()
print(A.addBinary('100','101'))


# In[9]:


#43.Multiply strings
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        return(str(int(num1)*int(num2)))
A=Solution()
print(A.multiply('100','99'))


# In[10]:


#50.Pow(X,n)
class Solution:
    def myPow(self, x: float, n: int) -> float:
             return(x**n)
A=Solution()
print(A.myPow(2,10))

    


# In[11]:


#860.Lemonade change
class Solution:
    def lemonadeChange(self, a: list[int]) -> bool:
                f=0
                te=0
                tw=0
                if a[0]>5:
                    return(False)
                else:    
                    for i in a:
                        if i==5:
                            f=f+1
                        elif i==10:
                            if f<1:
                                return(False)
                                break
                            else:    
                                f=f-1
                                te=te+1
                        else: 
                            if te >= 1 and f >= 1:
                                       te -= 1 
                                       f -= 1
                            elif f >= 3:
                                   f -= 3
                            else:
                                return False
                    else :
                        return(True) 
A=Solution()
print(A.lemonadeChange([5,5,5,10,20]))


# In[12]:


#1672.Richest customer wealth
class Solution:
    def maximumWealth(self, a: list[list[int]]) -> int:
        b=[]
        for i in a:
                b.append(sum(i))
        return(max(b)) 
A=Solution()
print(A.maximumWealth([[1,2,3],[5,7,8]]))


# In[13]:


#1572.Matrix diagonal sum
class Solution:
    def diagonalSum(self, a: list[list[int]]) -> int:
        c=0
        for i in range(len(a)):
                 c=c+a[i][i]+a[i][len(a)-i-1]
        if len(a)%2==0:
                 return(c)
        else:
            return(c-a[int(len(a)/2)][int(len(a)/2)])
A=Solution()
print(A.diagonalSum([[1,1,1,1],
              [1,1,1,1],
              [1,1,1,1],
              [1,1,1,1]]))


# In[14]:


#54.Spiral matrix
class Solution:
  def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
    if not matrix:
      return []

    m = len(matrix)
    n = len(matrix[0])
    ans = []
    r1 = 0
    c1 = 0
    r2 = m - 1
    c2 = n - 1

    # Repeatedly add matrix[r1..r2][c1..c2] to `ans`.
    while len(ans) < m * n:
      j = c1
      while j <= c2 and len(ans) < m * n:
        ans.append(matrix[r1][j])
        j += 1
      i = r1 + 1
      while i <= r2 - 1 and len(ans) < m * n:
        ans.append(matrix[i][c2])
        i += 1
      j = c2
      while j >= c1 and len(ans) < m * n:
        ans.append(matrix[r2][j])
        j -= 1
      i = r2 - 1
      while i >= r1 + 1 and len(ans) < m * n:
        ans.append(matrix[i][c1])
        i -= 1
      r1 += 1
      c1 += 1
      r2 -= 1
      c2 -= 1

    return ans
A=Solution()
print(A.spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))


# In[15]:


#73.Set matrix zeroes
class Solution:
    def setZeroes(self, a:list[list[int]]) -> None:
        b=[]
        c=[]
        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[i][j]==0:
                     b.append(i)
                     c.append(j)
        for i in b:
            for j in range(len(a[0])):
                a[i][j]=0
        for i in c:
            for j in range(len(a)):
                a[j][i]=0
A=Solution()
print(A.setZeroes([[0,1,2,0],[3,4,5,2],[1,3,1,5]]))


# In[16]:


#657.Robot return to origin
class Solution:
    def judgeCircle(self, moves: str) -> bool:
        a=list(moves)
        if (a.count('U')==a.count('D')) and (a.count('R')==a.count('L')):
            return(True)
        else:
            return(False)
A=Solution()
print(A.judgeCircle('UDLRD'))


# In[17]:


#1275.Tictactoe game
class Solution:
  def tictactoe(self, moves: list[list[int]]) -> str:
    row = [[0] * 3 for _ in range(2)]
    col = [[0] * 3 for _ in range(2)]
    diag1 = [0] * 2
    diag2 = [0] * 2
    i = 0

    for r, c in moves:
      row[i][r] += 1
      col[i][c] += 1
      diag1[i] += r == c
      diag2[i] += r + c == 2
      if 3 in (row[i][r], col[i][c], diag1[i], diag2[i]):
        return "A" if i == 0 else "B"
      i ^= 1

    return "Draw" if len(moves) == 9 else "Pending"
A=Solution()
print(A.tictactoe([[0,0],[2,0],[1,1],[2,1],[2,2]]))


# In[18]:


#1041.Robot bounded in circle
class Solution:
  def isRobotBounded(self, instructions: str) -> bool:
    x = 0
    y = 0
    d = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for instruction in instructions:
      if instruction == 'G':
        x += directions[d][0]
        y += directions[d][1]
      elif instruction == 'L':
        d = (d + 3) % 4
      else:
        d = (d + 1) % 4

    return (x, y) == (0, 0) or d > 0
A=Solution()
print(A.isRobotBounded("GGLLGG"))


# In[19]:


#1523.Count odd numbers in interval range
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        if (low%2==0) and (high%2==0):
            return(int((high-low)/2))
        elif (low%2!=0) and (high%2!=0):
            return(int(((high-low)/2)+1))
        else:
            return(int((int((high-low)/2)+1)))
A=Solution()
print(A.countOdds(23,70))


# In[ ]:




