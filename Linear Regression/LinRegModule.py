#!/usr/bin/env python
# coding: utf-8

# In[1]:


#module for Linear Regession
import numpy as np
import math


# In[3]:


class Lin_Reg:

    def fitt(self,Xin,Yin,alpha=0.1,i=10000,batch_size=0):
          self.X=np.insert(Xin,0,1,axis=1)  #inserting an extra column containing 1 in matrix X
          self.Y=Yin
          self.theta=np.random.rand(3,1)
          
          if(batch_size==0):
              batch_size=self.X.shape[0]//10   # Initializes the size of the mini batches if not provided by user
          
          self.alpha=alpha
          self.batch_size=batch_size
          self.i=i
          
          self.mini_gradient_descent()

    #Hypothesis function
    def hypothesis(self,X):
          H=np.dot(X,self.theta)
          return H
      
      
    #Cost function
    def cost(self):
          H=self.hypothesis(self.X)
          err=H-self.Y
          J = np.dot(err.T, err)/(2*self.X.shape[0])
          return J
      
    #gradient descent
    def gradient_des(self):
          m=self.X.shape[0]
          D=np.zeros((3,1))
          for j in range(self.itr):
              H=self.hypothesis(self.X)
              D=((H - self.Y).T.dot(self.X).T)/m
              self.theta = self.theta - D*self.alpha
          return self.theta

    #Again,functions like gradient descent BUT does so in small batches, hence named mini batch gradient descent
    def mini_gradient_descent(self):
          m=self.X.shape[0]
          r=math.ceil(m/self.batch_size)               # decides the no of mini batches
          
          for j in range(self.i):
              v=self.batch_size                        # manages overflowing
              for k in range(r):
                  u=k*self.batch_size
                  if(u+v>m):
                      v=m-u
                  X1=self.X[u:u+v,:]
                  Y1=self.Y[u:u+v,:]
                  
                  H=self.hypothesis(X1)
                  D=((H - Y1).T.dot(X1).T)/v
                  self.theta = self.theta - D*self.alpha
          return self.theta
    
    #Normalization for x
    def normalize(self,X):
          R=np.std(X,axis=0)
          M=np.mean(X,axis=0)
          X=(X-M)/R
          return X    
    
    #Calculates the accuracy of the prediction , always satisfying when it matches sklearn
    def accuracy(self,y_test,y_pred):
          err=(y_pred-y_test)*100/y_test
          return 100-np.mean(err) 

    #predictions on test data
    def predict(self,X):
          X=np.insert(X,0,1,axis=1)
          return self.hypothesis(X)
    


# In[ ]:




