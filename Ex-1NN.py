#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np


# In[3]:


df=pd.read_csv("Churn_Modelling.csv")


# In[4]:


df.head()
df.tail()
df.columns


# In[5]:


df.isnull().sum()


# In[6]:


X = df.iloc[:,:-1].values
X


# In[7]:


Y = df.iloc[:,-1].values
Y


# In[8]:


df.describe()


# In[9]:


data = df.drop(['Surname', 'Geography','Gender'], axis=1)


# In[10]:


data.head()


# In[11]:


scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)


# In[12]:


X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)


# In[13]:


X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))


# In[ ]:





# In[ ]:




