#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


class Log_Reg:
    
    def __init__(self, bsd_ftr, output, alpha, n_iter, b):
        self.x = bsd_ftr
        self.y = output
        self.alpha = alpha
        self.n_iter = n_iter
        self.m = len(output)
        self.n = len(bsd_ftr.T)
        self.b = b
        
    def sigmoid(self, theta):
        z = np.dot(self.x, theta)
        z = z.squeeze(1)
        hx = 1/(1+np.exp(-z))
        hx = hx.reshape(self.m, 1)
        return hx 
    
    #Regenerate output for 1 vs all 
    def yb(self):
        y1 = np.where(self.y==self.b, 'T', self.y)
        y2= np.where(y1!='T', float(0), y1)
        y3 = np.where(y2=='T', float(1), y2)
        y3 = y3.astype('float64')
        return y3

    def cost(self, theta):
        h = self.sigmoid(theta=theta)
        y = self.yb()
        cst = (-np.dot(self.y.T, np.log(h))-np.dot((1-self.y).T, np.log(1-h)))/self.m
        return cst

    def upd_theta(self, theta):
        hx = self.sigmoid(theta=theta)
        y = self.yb()
        drvtv = np.dot(self.x.T, (hx-y))/self.m
        theta = theta - self.alpha*drvtv
        return theta
    # PERFORMING ITERATIONS AND RETURN THETA 
    def fit(self, sq_error=np.array([]), I=np.array([])):
        theta = np.zeros(self.n).reshape(self.n, 1)
        for i in range(self.n_iter+1):
            i+=1
            theta = self.upd_theta(theta=theta)
            cst = self.cost(theta=theta)
            sq_error = np.append(sq_error, cst)
            I = np.append(I, i)
        list = ([plt.scatter(I, 1/sq_error, marker= ".", s=.2), theta, sq_error])
        return list
    
    #PREDICTED VALUE
    def predict(self, theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9):
        hx0=self.sigmoid(theta=theta0)
        hx1=self.sigmoid(theta=theta1)
        hx2=self.sigmoid(theta=theta2)
        hx3=self.sigmoid(theta=theta3)
        hx4=self.sigmoid(theta=theta4)
        hx5=self.sigmoid(theta=theta5)
        hx6=self.sigmoid(theta=theta6)
        hx7=self.sigmoid(theta=theta7)
        hx8=self.sigmoid(theta=theta8)
        hx9=self.sigmoid(theta=theta9)
        hx=np.hstack((hx0, hx1, hx2, hx3, hx4, hx5, hx6, hx7, hx8, hx9))
        return np.argmax(hx, axis=1).reshape(self.m, 1)
    
    #GETTINNG ACCURACY
    def accuracy(self, theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9):
        p=self.predict(theta0=theta0, theta1=theta1, theta2=theta2, theta3=theta3, theta4=theta4, theta5=theta5, theta6=theta6, theta7=theta7, theta8=theta8, theta9=theta9)
        acu = ((p==self.y).sum())*(100/self.m)
        return acu
    
    #FOR USING IT
    #TRAIN EVERY THETA DIFFERENTLY AS theta0=Log_Reg(n_iter, alpha, bsd_ftr=x1, output=y1, b=0).fit()[1]
    #ACCESS EVERY THETA FROM ABOVE CODE TO INPUT IT INTO FOR FINAL PREDICTION


# In[3]:



#ACCESSING TRAINING SET AND SPLIT IT INTO FEATUTRE MATRIX, OUTPUT MATRIX
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


# In[4]:


theta0=Log_Reg(n_iter=800, alpha=.0000003, bsd_ftr=x1, output=y1, b=0).fit()[1]
theta1=Log_Reg(n_iter=1000, alpha=.0000003, bsd_ftr=x1, output=y1, b=1).fit()[1]
theta2=Log_Reg(n_iter=700, alpha=.0000008, bsd_ftr=x1, output=y1, b=2).fit()[1]
theta3=Log_Reg(n_iter=1000, alpha=.0000003, bsd_ftr=x1, output=y1, b=3).fit()[1]
theta4=Log_Reg(n_iter=800, alpha=.0000005, bsd_ftr=x1, output=y1, b=4).fit()[1]
theta5=Log_Reg(n_iter=1000, alpha=.000003,  bsd_ftr=x1, output=y1, b=5).fit()[1]
theta6=Log_Reg(n_iter=1000, alpha=.0000003, bsd_ftr=x1, output=y1, b=6).fit()[1]
theta7=Log_Reg(n_iter=800, alpha=.0000003, bsd_ftr=x1, output=y1, b=7).fit()[1]
theta8=Log_Reg(n_iter=2000, alpha=.000008, bsd_ftr=x1, output=y1, b=8).fit()[1]
theta9=Log_Reg(n_iter=900, alpha=.000003,  bsd_ftr=x1, output=y1, b=9).fit()[1]


# In[5]:


c_inp = Log_Reg(bsd_ftr=x1, output=y1, n_iter=1000, alpha=.00000000004, b=0)
c_inp.accuracy(theta0=theta0, theta1=theta1, theta2=theta2, theta3=theta3, theta4=theta4, theta5=theta5, theta6=theta6, theta7=theta7, theta8=theta8, theta9=theta9)


# In[6]:


#ACCESSING TEST SET AND SPLIT IT INTO FEATURE MATRIX, OUTPUT MATRIX
df2 = pd.read_csv(r"F:\mnist_test.csv", header=None)
df2 = df2.values
y2 = df2[0:, 0:1]
features = df2[0:, 1:]
m2 = len(features)
n2= np.size(features, 1)
n2 = n2+1
bias = np.ones(m2).reshape(m2, 1)
x2= np.hstack((bias, features))


# In[7]:


c_out = Log_Reg(bsd_ftr=x2, output=y2, n_iter=1000, alpha=.00000000004, b=0)
c_out.accuracy(theta0=theta0, theta1=theta1, theta2=theta2, theta3=theta3, theta4=theta4, theta5=theta5, theta6=theta6, theta7=theta7, theta8=theta8, theta9=theta9)


# In[ ]:




