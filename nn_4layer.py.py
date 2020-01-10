#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[16]:


class NN:
    def __init__(self,ftr,y, s, lr, n_iter):
        self.m = len(y)
        self.a1=np.hstack((np.ones(self.m).reshape(self.m, 1), ftr))
        y = np.where(y==0, np.identity(10)[0], y) #RECREATING OUTPUT MATRIX FOR CALCULATING COST
        y = np.where(y==1, np.identity(10)[1], y)
        y = np.where(y==2, np.identity(10)[2], y)
        y = np.where(y==3, np.identity(10)[3], y)
        y = np.where(y==4, np.identity(10)[4], y)
        y = np.where(y==5, np.identity(10)[5], y)
        y = np.where(y==6, np.identity(10)[6], y)
        y = np.where(y==7, np.identity(10)[7], y)
        y = np.where(y==8, np.identity(10)[8], y)
        y = np.where(y==9, np.identity(10)[9], y)
        self.y = y
        self.s=s
        self.n = len(self.a1.T)
        self.lr=lr
        self.n_iter=n_iter
        #ALLOTING RANDOM VALUES TO WEIGHTS
        np.random.seed(0)
        self.w1=np.random.randn(self.n*s).reshape(self.n, s)
        np.random.seed(1)
        self.w2=np.random.randn((s)*s).reshape((s), s)
        np.random.seed(21)
        self.w3=np.random.randn((s)*s).reshape((s), s)
        np.random.seed(2)
        self.w4=np.random.randn((s)*10).reshape((s), 10)
        
    def cost(self,a5 ):
            l =0
            for i in range(10):
                l=(np.dot(self.y[:, i:i+1].T, np.log(a5[:, i:i+1])) + np.dot((1-self.y)[:, i:i+1].T, np.log((1-a5)[:, i:i+1])))+l
            return(l*(-1/self.m))
        
    def sigmoid(self, z, drvtv=False):  #IF DRVTV=tRUE THEN GIVING DRVTV
        if (drvtv==True):
            return z*(1-z)
        return 1/(1+np.exp(-z))
    
    #FORWARD PROPAGATION FOR FINDING PREDICTED VALUE
    def fwd(self, w1, w2, w3, w4):
        self.z2=np.dot(self.a1, w1)
        self.a2=self.sigmoid(z=self.z2)
        #self.a2=np.hstack((np.ones(self.m).reshape(self.m, 1), self.a2))
        self.z3=np.dot(self.a2, w2)
        self.a3=self.sigmoid(z=self.z3)
        #self.a3=np.hstack((np.ones(self.m).reshape(self.m, 1), self.a3))
        self.z4=np.dot(self.a3, w3)
        self.a4=self.sigmoid(z=self.z4)
        #self.a4=np.hstack((np.ones(self.m).reshape(self.m, 1), self.a4))
        self.z5=np.dot(self.a4, w4)
        self.a5=self.sigmoid(z=self.z5)
        return ([self.a2, self.a3, self.a4, self.a5])
    
    #PERFORMING BACKWARD PROP. TO UPDATE WEIGHTS
    def back(self, w1, w2, w3, w4):
        a5=self.fwd(w1=w1, w2=w2, w3=w3, w4=w4)[3]
        a4=self.fwd(w1=w1, w2=w2, w3=w3, w4=w4)[2]
        a3=self.fwd(w1=w1, w2=w2, w3=w3, w4=w4)[1]
        a2=self.fwd(w1=w1, w2=w2, w3=w3, w4=w4)[0]
        a1=self.a1
        
        d5=a5-self.y
        d4=np.dot(d5, w4.T)*self.sigmoid(z=a4, drvtv=True)
        d3=np.dot(d4, w3.T)*self.sigmoid(z=a3, drvtv=True)
        d2=np.dot(d3, w2)*self.sigmoid(z=a2, drvtv=True)
        
        grad4=np.dot(a4.T, d5)/self.m
        grad3=np.dot(a3.T, d4)/self.m
        grad2=np.dot(a2.T, d3)/self.m
        grad1=np.dot(a1.T, d2)/self.m
        
        w4=w4-(self.lr)*grad4
        w3=w3-(self.lr)*grad3
        w2=w2-(self.lr)*grad2
        w1=w1-(self.lr)*grad1
        
        list=([w1, w2, w3, w4, a5])
        return list
    
    #DOING ITERATIONS TO GIVE FINAL WEIGHTS
    def fit(self, sq_error=np.array([]), I=np.array([])):
        w1=self.w1
        w2=self.w2
        w3=self.w3
        w4=self.w4
        for i in range(self.n_iter+1):
            i+=1
            w=self.back(w1=w1, w2=w2, w3=w3, w4=w4)
            w1=w[0]
            w2=w[1]
            w3=w[2]
            w4=w[3]
            a5=w[-1]
            cst=self.cost(a5=a5)
            sq_error = np.append(sq_error, cst)
            I = np.append(I, i) 
        a5=self.back(w1=w1, w2=w2, w3=w3, w4=w4)[4]
        list = ([plt.scatter(I, sq_error, marker= ".", s=.52), w1, w2, w3, w4, sq_error,  a5])
        return list   
            
    def out(self, a5):
        a5=np.around(a5)
        a5=np.argmax(a5, axis=1).reshape(self.m, 1)
        return a5
    
    def accuracy(self, a5 ):
        p=self.out(a5=a5)
        y=np.argmax(self.y, axis=1).reshape(self.m, 1)
        acu = ((p==y).sum())*(100/self.m)
        return acu


# In[17]:


df = pd.read_csv(r"F:\mnist_train_small.csv", header=None)
df.head()
df = df.values
y1 = df[0:, 0:1]
x1 = df[0:, 1:]


# In[18]:


#finding weights
c=NN(ftr=x1, y=y1, s=36, lr=4, n_iter=1000)
aa=c.fit()


# In[19]:


df2 = pd.read_csv(r"F:\mnist_test.csv", header=None)
df2 = df2.values
y2 = df2[0:, 0:1]
x2 = df2[0:, 1:]


# In[20]:


#training accuracy
c=NN(ftr=x2, y=y2, s=48, lr=4, n_iter=1000)
w1=aa[1]
w2=aa[2]
w3=aa[3]
w4=aa[4]
a=c.fwd(w1=w1, w2=w2, w3=w3, w4=w4)[3]
a=c.accuracy(a5=a)
a


# In[21]:


#test accuracy
c=NN(ftr=x1, y=y1, s=48, lr=4, n_iter=1000)
w1=aa[1]
w2=aa[2]
w3=aa[3]
w4=aa[4]
a=c.fwd(w1=w1, w2=w2, w3=w3, w4=w4)[3]
a=c.accuracy(a5=a)
a


# In[ ]:




