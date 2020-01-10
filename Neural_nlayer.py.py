#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


class NN:
    def __init__(self,ftr,y, size, lr, n_iter):  #size=[s2, s3, ..., 10] where sl is no. of units in hidden layer l
        self.m = len(y)
        self.a1=np.hstack((np.ones(self.m).reshape(self.m, 1), ftr))
        self.size=np.insert(size, 0, len(self.a1.T))
        for i in range(10):
            y = np.where(y==i, np.identity(10)[i], y)  ##RECREATING OUTPUT MATRIX FOR CALCULATING COST
            self.y = y
        self.lr=lr
        self.n_iter=n_iter
    
    ##ALLOTING RANDOM VALUES TO WEIGHTS
    def weights(self):
        W=[]
        for i in range(len(self.size)-1):
            np.random.seed(i)
            w=np.random.randn((self.size[i])*(self.size[(i+1)])).reshape((self.size[i]), (self.size[(i+1)]))
            W.append(w)
            a=np.asarray(W)
        return a
    
    def cost(self,a_last ):
        l =0
        for i in range(10):
            l=(np.dot(self.y[:, i:i+1].T, np.log(a_last[:, i:i+1])) + np.dot((1-self.y)[:, i:i+1].T, np.log((1-a_last)[:, i:i+1])))+l
        return(l*(-1/self.m))    
        
    def sigmoid(self, z, drvtv=False):
        if (drvtv==True):
            return z*(1-z)
        return 1/(1+np.exp(-z))
    
    #FORWARD PROPAGATION FOR FINDING PREDICTED VALUE
    def fwd(self, W):
        A=np.array([])
        A=np.append(A, self.a1)
        for i in range((len(self.size)-1)):
            z=np.dot(A[i], W[i])
            A=np.append(A,self.sigmoid(z=z))
        return A
    
    #PERFORMING BACKWARD PROP. TO UPDATE WEIGHTS
    def back(self, W):
        A=self.fwd(W=W)
        m=len(self.size)-2
        D=np.array([])
        d_last=(A[-1]-self.y)*self.sigmoid(z=A[-1], drvtv=True)
        D=np.append(D,d_last)
        
        for i in range(m):
            d=np.dot(D[-1], W[m-i].T)*self.sigmoid(z=A[m-i], drvtv=True)
            D=np.append(D,d)
        G=np.array([])
        for i in range(m+1):
            grad=np.dot(A[i].T, D[m-i])/self.m
            G=np.append(G,grad)                           #G=[grad1, grad2,..., grad_last]
       # W=np.asarray(W)
      #  G=np.asarray(G)
        W=W-self.lr*G
        W=np.append(W, A[-1])
        #W.append(A[-1])  #W=[w1, w2,.., w_last, a_last]
        return W    
    
    #DOING ITERATIONS TO GIVE FINAL WEIGHTS
    def fit(self, sq_error=np.array([]), I=np.array([])):
        W=self.weights()
        for i in range(self.n_iter+1):
            i+=1
            w_a=self.back(W=W)
            w=w_a[0:-1]
            a_last=w_a[-1]
            cst=self.cost(a_last=a_last)
            sq_error = np.append(sq_error, cst)
            I = np.append(I, i)
            a_last=self.back(W=w)[-1]
            list = ([plt.scatter(I, sq_error, marker= ".", s=.52), W, sq_error,  a_last])
        return list
    
    def out(self, a_last):
        a_last=np.around(a_last)
        a_last=np.argmax(a_last, axis=1).reshape(self.m, 1)
        return a_last
    
    def accuracy(self, a_last ):
        p=self.out(a_last=a_last)
        y=np.argmax(self.y, axis=1).reshape(self.m, 1)
        acu = ((p==y).sum())*(100/self.m)
        return acu


# In[8]:


df = pd.read_csv(r"F:\mnist_train_small.csv", header=None)
df.head()
df = df.values
y1 = df[0:, 0:1]
x1 = df[0:, 1:]
df2 = pd.read_csv(r"F:\mnist_test.csv", header=None)
df2 = df2.values
y2 = df2[0:, 0:1]
x2 = df2[0:, 1:]


# In[20]:


#finding FINAL weights
c=NN(ftr=x1, y=y1, size=[16, 16, 10], lr=4, n_iter=10)
aa=c.fit()
aa


# In[ ]:


c=NN(ftr=x1, y=y1, size=[16, 16, 10], lr=4, n_iter=10)
W=aa[0:-1]
a=c.fwd(W=W)[-1]
a=c.accuracy(a_last=a)
a


# In[ ]:




