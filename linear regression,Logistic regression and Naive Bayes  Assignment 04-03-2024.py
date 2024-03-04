#!/usr/bin/env python
# coding: utf-8

# In[77]:


from sklearn.datasets import load_breast_cancer,load_diabetes##import neccessary libraries and functions
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt


# ###### Linear Regression in python

# In[78]:



diabetes=load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Add the target variable to the dataframe
df['target'] = diabetes.target


# In[79]:


print(df.head())


# In[80]:


df.describe()


# In[81]:


df.isna().sum()


# In[82]:


X=df.drop(columns=['target'])
Y=df['target']


# In[ ]:





# In[83]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)#create train and test datasets for model prediction


# In[84]:


Y_train.shape


# In[85]:


Y_test.shape


# In[86]:


Y_train=Y_train.values.reshape((353,1))
Y_test=Y_test.values.reshape((89,1))


# X_train1 = X_train.values#Final train and test datasets for model prediction
# Y_train1 = Y_train.values
# X_test1 = X_test.values
# Y_test1= Y_test.values

# In[87]:


print("Shape of X_train : ", X_train.shape)
print("Shape of Y_train : ", Y_train.shape)
print("Shape of X_test : ", X_test.shape)
print("Shape of Y_test : ", Y_test.shape)


# In[88]:


Y=Y.values.reshape((442,1))


# X = np.vstack((np.ones((X.shape[ 0 ], )), X.T)).T
# X_test = np.vstack((np.ones((X_test.shape[ 0 ], )), X_test.T)).T

# In[89]:


def model(X, Y, learning_rate, iteration):
        m = Y.size
        theta = np .zeros((X.shape[ 1 ], 1 ))
        cost_list = []
        for i in range (iteration):
                y_pred = np .dot(X, theta)
                cost = ( 1 /( 2 *m))* np . sum ( np .square(y_pred - Y))
                d_theta = ( 1 /m)* np .dot(X.T, y_pred - Y)
                theta = theta - learning_rate*d_theta
                cost_list. append (cost)
        # to print the cost for 10 times
                if (i%(iteration/ 10 ) == 0 ):
                        print ( "Cost is :" , cost)
        return theta, cost_list


# In[90]:


iteration = 10000
learning_rate = 0.0000005
theta, cost_list = model(X, Y, learning_rate = learning_rate, iteration =iteration)


# In[91]:


rng = np.arange( 0 , iteration)
plt. plot (rng, cost_list)
plt. show ()


# In[92]:


y_pred = np.dot(X_test, theta)
error = (1/X_test.shape[0])*np.sum(np.abs(y_pred - Y_test))


# In[93]:


print ( "Test error is :" , error *100 , "%" )
print ( "Test Accuracy is :" , (1- error) *100 , "%" )


# ###### Logistic regression

# In[94]:


breast_cancer=load_breast_cancer()

df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# Add the target variable to the dataframe
df['target'] = breast_cancer.target




# In[95]:


print(df.head())


# In[96]:


df.describe()


# In[97]:


df.isna().sum()


# In[98]:


X=df.drop(columns=['target'])
Y=df['target']


# In[99]:


X


# In[100]:


Y


# In[101]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)#create train and test datasets for model prediction


# In[102]:


X_train1 = X_train.values#Final train and test datasets for model prediction
Y_train1 = Y_train.values
X_test1 = X_test.values
Y_test1= Y_test.values


# In[103]:


X_train1.shape


# In[104]:


X_train1=X_train.T


# In[105]:


X_train1.shape


# In[106]:


Y_train1 = Y_train1.reshape(1, X_train1.shape[1])


# In[107]:


X_test1 = X_test1.T


# In[108]:


Y_test1 = Y_test1.reshape(1, X_test1.shape[1])


# In[109]:


print("Shape of X_train : ", X_train1.shape)
print("Shape of Y_train : ", Y_train1.shape)
print("Shape of X_test : ", X_test1.shape)
print("Shape of Y_test : ", Y_test1.shape)


# # Logistic regression

# In[110]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[111]:


def model(X, Y, learning_rate, iterations):
    
    m = X_train1.shape[1]
    n = X_train1.shape[0]
    
    W = np.zeros((n,1))
    B = 0
    
    cost_list = []
    
    for i in range(iterations):
        
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        
        # cost function
        cost = -(1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A))
        
        # Gradient Descent
        dW = (1/m)*np.dot(A-Y, X.T)
        dB = (1/m)*np.sum(A - Y)
        
        W = W - learning_rate*dW.T
        B = B - learning_rate*dB
        
        # Keeping track of our cost function value
        cost_list.append(cost)
        
        if(i%(iterations/10) == 0):
            print("cost after ", i, "iteration is : ", cost)
        
    return W, B, cost_list
        


# In[112]:


iterations = 1000
learning_rate = 0.0012
W, B, cost_list = model(X_train1, Y_train1, learning_rate = learning_rate, iterations = iterations)


# ###### Cost vs Iteration
# Plotting graph to see if Cost Function is decreasing or not

# In[113]:


plt.plot(np.arange(iterations), cost_list)
plt.show()


# ###### Testing Model Accuracy

# In[114]:


def accuracy(X, Y, W, B):
    
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    
    A = A > 0.5
    
    A = np.array(A, dtype = 'int64')
    
    acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
    
    print("Accuracy of the model is : ", round(acc, 2), "%")
    


# In[115]:


accuracy(X_test1, Y_test1, W, B)


# # Naive bayes 

# ###### Calculate P(Y=y) for all possible y

# In[116]:


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior


# ###### Calculate P(X=x|Y=y) using Gaussian dist.

# In[117]:


def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y


# ###### Calculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y and find the maximum

# In[118]:


def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)


# ###### Test Gaussian model

# In[119]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=.2, random_state=41)

X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train, X=X_test, Y="target")

from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))

