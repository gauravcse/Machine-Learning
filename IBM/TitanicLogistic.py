
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:

data = pd.read_csv('titanic_train.csv',index_col = 0)


# In[3]:

data.head()


# In[4]:

y = data.iloc[:,[0]]
sex = data.iloc[:,[3]]
sex.replace(to_replace={"male","female"},value={0,1},inplace=True)
age = data.iloc[:,[4]]
fare = data.iloc[:,[8]]
sib_sp = data.iloc[:,[5]]
embarked = data.iloc[:,[10]]
embarked.replace(to_replace={"S","C","Q"},value={0,1,2},inplace=True)
pclass = data.iloc[:,[1]]


# In[5]:

x_train = pd.concat([sex,age,fare,sib_sp,embarked,pclass],axis = 1)


# In[6]:

x_train = x_train.fillna(x_train.median())
x_train.head()


# In[7]:

from sklearn.cross_validation import train_test_split


# In[15]:

X_train,X_test,y_train,y_test = train_test_split(x_train,y,random_state = 15)


# In[9]:

from sklearn.linear_model import LogisticRegression


# In[16]:

nb = LogisticRegression()
nb.fit(X_train,y_train)


# In[17]:

y_pred = nb.predict(X_test)


# In[18]:

from sklearn.metrics import accuracy_score


# In[19]:

err = accuracy_score(y_test,y_pred)


# In[20]:

print err


# In[21]:

nb1 = LogisticRegression()
nb1.fit(x_train,y)


# In[22]:

data_t = pd.read_csv('titanic_test.csv',index_col = 0)
data_t.head()


# In[23]:

sex_y = data_t.iloc[:,[2]]
sex_y.replace(to_replace={"male","female"},value={0,1},inplace=True)
age_y = data_t.iloc[:,[3]]
fare_y = data_t.iloc[:,[7]]
sib_sp_y = data_t.iloc[:,[4]]
embarked_y = data_t.iloc[:,[9]]
embarked_y.replace(to_replace={"S","C","Q"},value={0,1,2},inplace=True)
pclass_y = data_t.iloc[:,[0]]


# In[24]:

x_test = pd.concat([sex_y,age_y,fare_y,sib_sp_y,embarked_y,pclass_y],axis = 1)
x_test.head()
x_test = x_test.fillna(x_test.median())


# In[25]:

y_pred_test = nb1.predict(x_test)
y_pred_t = pd.Series(y_pred_test)


# In[26]:

y_index = (pd.read_csv('titanic_test.csv')).iloc[:,[0]]


# In[27]:

output = pd.concat([y_index,y_pred_t],axis = 1)


# In[28]:

output.head()


# In[29]:

output.to_csv('titanic_output_logit.csv')


# In[ ]:



