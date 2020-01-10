#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# In[3]:


class KNN:
    def __init__(self, training, test, k):
        self.k=k
        self.training=training
        self.test=test
        
    def Euclidean(self, row1, row2):  #Distance between points
        m=row1-row2
        m=(np.square(m)).sum()
        m=np.sqrt(m)
        return m

    def sortlast(self,val):   #To sort rows of array by arranging elements of column
        return val[-1]
    
    def neighbours(self, test_elmnt):  #GETTING NEIGHBOURS
        distances=[]
        for x in range(len(self.training)):
            dist=self.Euclidean(row1=self.training[x][1:], row2=test_elmnt[1:])
            distances.append((self.training[x],dist))
        distances.sort(key= self.sortlast)   
        nbr=[]
        for x in range(self.k):
            nbr.append(distances[x][0])
        return nbr    
    
    def response(self, test_elmnt):  #GIVING US CORRESPONDING CLASS FROM WHICH GIVEN TEST ELEMENT BELONGS 
        n=self.neighbours(test_elmnt=test_elmnt)
        clas=[]
        for x in range(self.k):
            clas.append(n[x][0])
        m=stats.mode(clas)   
        return m[0]
    
    def predict(self):
        y=[]
        for i in range(len(self.test)):
            y.append(self.response(test_elmnt=self.test[i]))
        return y
    
    def test_accuracy(self ):
        p=self.predict()
        #p = np.around(p)
        acu = ((p==self.test[0:, 0:1]).sum())*(100/len(self.test))
        return acu


# In[4]:


df = pd.read_csv(r"F:\mnist_train_small.csv", header=None)
df.head()
df = df.values
df2 = pd.read_csv(r"F:\mnist_test.csv", header=None)
df2 = df2.values


# In[6]:


c=KNN(training=df, test=df2[0:1000], k=5)
c.test_accuracy()


# In[7]:


c=KNN(training=df, test=df2[1000:2000], k=5)
c.test_accuracy()


# In[8]:


c=KNN(training=df, test=df2[2000:3000], k=5)
c.test_accuracy()


# In[9]:


c=KNN(training=df, test=df2[3000:4000], k=5)
c.test_accuracy()


# In[ ]:


c=KNN(training=df, test=df2[4000:5000], k=5)
c.test_accuracy()


# In[ ]:


c=KNN(training=df, test=df2[5000:6000], k=5)
c.test_accuracy()


# In[ ]:


c=KNN(training=df, test=df2[6000:7000], k=5)
c.test_accuracy()


# In[ ]:


c=KNN(training=df, test=df2[7000:8000], k=5)
c.test_accuracy()


# In[ ]:


c=KNN(training=df, test=df2[8000:9000], k=5)
c.test_accuracy()


# In[ ]:


c=KNN(training=df, test=df2[9000:], k=5)
c.test_accuracy()


# In[18]:


total_accuracy = 95.91

