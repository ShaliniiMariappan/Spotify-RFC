#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings


# In[3]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv("spotify_long_tracks_2014_2024.csv")
df


# In[5]:


from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
df['ID']=lb.fit_transform(df['ID'])
df['Name']=lb.fit_transform(df['Name'])
df['Artists']=lb.fit_transform(df['Artists'])
df['Duration (Minutes)']=lb.fit_transform(df['Duration (Minutes)'])


# In[6]:


df.info()


# In[7]:


x=df.iloc[:,1:2].values
x


# In[8]:


y=df.iloc[:,2].values
y


# In[9]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1, random_state=0)  


# In[10]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)   


# In[11]:


from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 5, criterion="entropy") 


# In[12]:


classifier.fit(x_train, y_train) 


# In[13]:


y_pred= classifier.predict(x_test) 


# In[14]:


y_pred


# In[15]:


from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  


# In[16]:


cm


# In[17]:


ylala=classifier.predict([[608]])
ylala


# In[18]:


clf = RandomForestClassifier(n_estimators = 100) 


# In[20]:


clf.fit(x_train, y_train)


# In[21]:


from sklearn import metrics  
print()


# In[22]:


print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))


# In[24]:


clf.predict([[3]])


# In[ ]:





# In[ ]:




