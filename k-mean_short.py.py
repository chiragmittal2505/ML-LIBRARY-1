#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"F:\mnist_train_small.csv", header=None)
df.head()
df = df.values
y1 = df[0:, 0:1]
x1 = df[0:, 1:]
df2 = pd.read_csv(r"F:\mnist_test.csv", header=None)
df2 = df2.values
y2 = df2[0:, 0:1]
x2 = df2[0:, 1:]


# In[3]:


class k_means:
    def __init__(self,df, k, n_iter):
        self.k=k
        self.df=df
        self.m=len(df)
        self.n=len(df.T)
        self.n_iter=n_iter
        
    def Euclidean(self, row1, row2):
        m=row1-row2
        m=(np.square(m)).sum()
        m=np.sqrt(m)
        return m
    
    def choose_centroid(self):
        K=[]
        for i in range(self.k):
            np.random.seed(i)
            k_=np.random.randn(self.n)
            K.append(k_)
        return K
    
    def predict(self, K):
        A=np.array([])
        for i in range(self.m):
            B=np.array([])
            for j in range(self.k):
                d=self.Euclidean(row1=self.df[i], row2=K[j])
                B=np.append(B,d)
            A=np.append(A, B)    
        n=np.asarray(A).reshape(self.m, self.k)
        n=n.reshape(self.m, self.k)
        t=np.argmin(n, axis=1).reshape(self.m, 1)
        #r=np.hstack((self.df, t)) 
        return t
    
    def allot_cluster(self, K):
        t=self.predict(K=K)
        H2=t[0:, -1]
        H2=H2.reshape(self.m, 1)
        Class=np.array([])
        for i in range(self.k):
            c=np.where(H2==i)[0]
            clas=self.df[c]
            C=np.append(Class, clas)
        
            return Class
    
    def update_centroid(self,K):
        class_=self.allot_cluster(K=K)
        K_=[]
        for i in range(len(class_)):
            n=len(class_[i])+1
            a=class_[i]
            b=K[i]
            j=np.vstack((a, b))
            j=j.sum(axis=0)/n
            K_.append(j)
        return K_    
        
    def fit(self):
        K=self.choose_centroid()
        for i in range(self.n_iter+1):
            i+=1
            K=self.update_centroid(K=K)
        return K
    
    def accuracy(self, K, ):
        p=self.predict(K=K)
        y=self.df[0]
        acu = ((p==y).sum())*(100/self.m)
        return acu


# In[4]:


c=k_means(df=df, k=10, n_iter=10)
K=c.fit()
K


# In[5]:


c.allot_cluster(K=K)


# In[ ]:




