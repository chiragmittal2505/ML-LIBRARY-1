#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


#ACCESSING DATAFRAMES
df = pd.read_csv(r"F:\mnist_train_small.csv", header=None)
df.head()
df = df.values
df2 = pd.read_csv(r"F:\mnist_test.csv", header=None)
df2 = df2.values


# In[12]:


class k_means:
    def __init__(self,df, k, n_iter):
        self.k=k
        self.df=df
        self.m=len(df)
        self.n=len(df.T)
        self.n_iter=n_iter
        
    # DISTANCES BETWEEN POINTS
    def Euclidean(self, row1, row2):
        m=row1-row2
        m=(np.square(m)).sum()
        m=np.sqrt(m)
        return m
    
    # RANDOM SELECTION OF CENTROID
    def choose_centroid(self):
        K=[]
        for i in range(self.k):
            np.random.seed(i)
            k_=np.random.randn(self.n)
            K.append(k_)
        return K
    
    #IT GIVES CLASS CORRESPONDING TO EXAMPLES
    def predict(self, K):
        A=np.array([])
        for i in range(self.m):
            B=np.array([])
            for j in range(self.k):
                d=self.Euclidean(row1=self.df[i][1:], row2=K[j][1:])
                B=np.append(B,d)
            A=np.append(A, B)    
        n=np.asarray(A).reshape(self.m, self.k)
        n=n.reshape(self.m, self.k)
        t=np.argmin(n, axis=1).reshape(self.m, 1)
        #r=np.hstack((self.df, t)) 
        return t
    
    #FORMING NEW CLUSTERS
    def allot_cluster(self, K):
        t=self.predict(K=K)
        H2=t[0:, -1]
        H2=H2.reshape(self.m, 1)
    
        c0=np.where(H2==0)[0]
        c1=np.where(H2==1)[0]
        c2=np.where(H2==2)[0]
        c3=np.where(H2==3)[0]
        c4=np.where(H2==4)[0]
        c5=np.where(H2==5)[0]
        c6=np.where(H2==6)[0]
        c7=np.where(H2==7)[0]
        c8=np.where(H2==8)[0]
        c9=np.where(H2==9)[0]
        class0=self.df[c0]
        class1=self.df[c1]
        class2=self.df[c2]
        class3=self.df[c3]
        class4=self.df[c4]
        class5=self.df[c5]
        class6=self.df[c6]
        class7=self.df[c7]
        class8=self.df[c8]
        class9=self.df[c9]
        class_ =np.array([class0, class1, class2, class3, class4, class5, class6, class7 ,class8, class9])
        return class_
    
    #UPDATING CENTROID
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


# In[13]:


#GETTING VALUE OF RANDOM CENTROIDS
c=k_means(df=df, k=10, n_iter=100)
K=c.fit()
K


# In[14]:


#OBTAINING CLUSTERS
c.allot_cluster(K=K)

