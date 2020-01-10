#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


class Lin_Reg:
    
    def __init__(self, bsd_ftr, output, alpha, n_iter):   #input: biased feaatures, output, learning rate, iterations
        self.x = bsd_ftr
        self.y = output
        self.alpha = alpha
        self.n_iter = n_iter
        self.m = len(output)
        self.n = len(bsd_ftr.T)
        
    def predict(self, theta):                 #gives predicted value
        hx = np.dot(self.x, theta)
        return hx
    
    def cost(self, theta):                  
        hx = self.predict(theta=theta)
        error = hx - self.y
        cst = np.dot(error.T, error)/(2*self.m)
        return cst
    
    def upd_theta(self, theta):
        hx = self.predict(theta=theta)
        error = hx - self.y
        grad = np.dot(self.x.T, error)
        theta = theta - self.alpha*grad
        return theta
    
    def fit(self, sq_error=np.array([]), I=np.array([])):   #perform no. of iterations to return final theta, mean sq. error
        np.random.seed(42)
        theta = np.zeros(self.n).reshape(self.n, 1)
        for i in range(self.n_iter+1):
            i+=1
            theta = self.upd_theta(theta=theta)
            cst = self.cost(theta=theta)
            sq_error = np.append(sq_error, cst)
            I = np.append(I, i)
        list = ([plt.scatter(I, sq_error, marker= ".", s=.2), theta, sq_error])
        return list
    
    def accuracy(self, theta ):
        p=self.predict(theta=theta)
        p = np.around(p)
        acu = ((p==self.y).sum())*(100/self.m)
        return acu

     #use   self.fit()[1] for accessing theta


# In[13]:


df = pd.read_csv(r"F:\mnist_train_small.csv", header=None)
df.head()
df = df.values
y1 = df[0:, 0:1]
features = df[0:, 1:]
m = len(features)
n = np.size(features, 1)
n = n+1
bias = np.ones(m).reshape(m, 1)
x1 = np.hstack((bias, features))


# In[14]:


c_inp = Lin_Reg(bsd_ftr=x1, output=y1, n_iter=2000, alpha=.00000000004)
theta0=c_inp.fit()[1]
c_inp.accuracy(theta=theta0)


# In[15]:


df2 = pd.read_csv(r"F:\mnist_test.csv", header=None)
df2 = df2.values
y2 = df2[0:, 0:1]
features = df2[0:, 1:]
m2 = len(features)
n2= np.size(features, 1)
n2 = n2+1
bias = np.ones(m2).reshape(m2, 1)
x2= np.hstack((bias, features))


# In[16]:


c_out = Lin_Reg(bsd_ftr=x2, output=y2, n_iter=2500, alpha=.00000000004)
c_out.accuracy(theta=theta0)


# In[ ]:




