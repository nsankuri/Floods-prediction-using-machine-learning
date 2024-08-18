#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('C:/Users/nihar/Downloads/project dataset.csv')
df.head()


# Finding number of missing values
# 

# In[4]:


df.isnull().sum()


# In[5]:


print(df.shape)


# In[6]:


df.describe()


# In[7]:


df['FLOODS'].replace(['YES','NO'],[1,0],inplace=True)


# In[8]:


df.drop('SUBDIVISION',axis = 1,inplace=True)


# In[9]:


df.head()


# #**4. Analyze**

# In[10]:


x = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[11]:


x.head()


# In[12]:


y.head()


# *Let's see how the rainfall index varies during the rainy season, usually from  June to September.* 

# In[14]:


AvgIndex = df[['JUN','JUL','AUG','SEP']]
AvgIndex.hist()
plt.show()


# *Data is widely distributed , let's scale it down to 0 and 1*

# In[13]:


from sklearn import preprocessing
minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit_transform(x)


# In[14]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)


# #Prediction Algorithms:
# 

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[16]:


def mymodel(model):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    
    train = model.score(xtrain, ytrain)
    test = model.score(xtest, ytest)
    
    print(f"Training Accuracy : {train}\nTesting Accuracy : {test}\n\n")
    print(classification_report(ytest, ypred))
    
    return model


# ##*1. KNN Alogrithm*
# 
# ---
# 
# 

# In[17]:


knn = mymodel(KNeighborsClassifier())


# ##*2. Logistic Regression Alogrithm*
# 
# ---
# 
# 

# In[18]:


logreg = mymodel(LogisticRegression())


# ##*3. Support Vector Machine Alogrithm*
# 
# ---
# 
# 

# In[19]:


svm = mymodel(SVC())


# ##*4. Decision Tree Alogrithm*
# 
# ---
# 
# 

# In[20]:


dt = mymodel(DecisionTreeClassifier())


# #Cross-validation
# 

# In[21]:


from sklearn.model_selection import cross_val_score


# In[22]:


knn_accuracy = cross_val_score(knn,xtest,ytest,cv=3,scoring='accuracy',n_jobs=-1)
knn_accuracy.mean()


# In[23]:


logreg_accuracy = cross_val_score(logreg,xtest,ytest,cv=3,scoring='accuracy',n_jobs=-1)
logreg_accuracy.mean()


# In[24]:


svm_accuracy = cross_val_score(svm,xtest,ytest,cv=3,scoring='accuracy',n_jobs=-1)
svm_accuracy.mean()


# In[25]:


dt_accuracy = cross_val_score(dt,xtest,ytest,cv=3,scoring='accuracy',n_jobs=-1)
dt_accuracy.mean()


# In[26]:


names = ['KNN','LogReg','SVM','DecisionTree']
score =[knn_accuracy.mean(),logreg_accuracy.mean(),svm_accuracy.mean(),dt_accuracy.mean()]


# In[27]:


scores = pd.DataFrame({'Algorithm Name':names,'Score':score})


# In[28]:


axis = sns.barplot(x='Algorithm Name',y='Score',data = scores)
axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    


# In[29]:


scores # Cross-validation


# #2nd Models 

# ##HyperParameter Tuning For Decsion Tree

# In[30]:


dt = mymodel(DecisionTreeClassifier()) # Current Accuracy


# # *GridSearchCV*

# In[31]:


parameters = {
    "criterion":["gini", "entropy"],
    "max_depth": list(range(1,50, 5)),
    "min_samples_leaf": list(range(1, 50, 5))
}


# In[32]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(DecisionTreeClassifier(), parameters, verbose=2)
grid.fit(xtrain, ytrain)


# In[33]:


grid.best_params_


# In[34]:


grid.best_score_


# In[35]:


grid.best_estimator_


# In[36]:


dt2 = mymodel(grid.best_estimator_) # Post- HyperParameter Tuning For Decision Tree


# ##HyperParameter Tuning For SVM

# # *GridSearchCV*

# In[37]:


parameters = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}


# In[38]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), parameters, verbose=3)
grid.fit(xtrain, ytrain)


# In[39]:


grid.best_params_


# In[40]:


grid.best_score_


# In[41]:


grid.best_estimator_


# In[42]:


svm = mymodel(grid.best_estimator_)


# ##HyperParameter Tuning For LogReg
# 
# # *GridSearchCV*

# In[43]:


parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}


# In[44]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LogisticRegression(), parameters, verbose=2)
grid.fit(xtrain, ytrain)


# In[45]:


grid.best_params_


# In[46]:


grid.best_score_


# In[47]:


grid.best_estimator_


# In[48]:


logreg = mymodel(grid.best_estimator_)


# ##HyperParameter Tuning For KNN
# 
# # *GridSearchCV*

# In[49]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline(
    [
        ("sc", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ]
)


# In[50]:


from sklearn.model_selection import GridSearchCV
parameters = [{"knn__n_neighbors": [3, 5, 7, 9],
              "knn__weights": ["uniform", "distance"],
              "knn__leaf_size": [15, 20]}]


# In[51]:


grid = GridSearchCV(pipe, parameters, cv=5, scoring="accuracy")
grid.fit(xtrain, ytrain)


# In[52]:


grid.best_params_


# In[53]:


grid.best_score_


# In[54]:


grid.best_estimator_


# In[55]:


knn2 = mymodel(grid.best_estimator_)


# In[56]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.3, random_state=1, stratify=y)


# In[57]:


knn_accuracy = cross_val_score(knn2,xtest,ytest,cv=3,scoring='accuracy')
knn_accuracy.mean()


# In[58]:


logreg_accuracy = cross_val_score(logreg,xtest,ytest,cv=3,scoring='accuracy',n_jobs=-1)
logreg_accuracy.mean()


# In[59]:


svm_accuracy = cross_val_score(svm,xtest,ytest,cv=3,scoring='accuracy',n_jobs=-1)
svm_accuracy.mean()


# In[60]:


dt_accuracy = cross_val_score(dt,xtest,ytest,cv=3,scoring='accuracy',n_jobs=-1)
dt_accuracy.mean()


# In[61]:


names = ['KNN','LogReg','SVM','DecisionTree']
score =[knn_accuracy.mean(),logreg_accuracy.mean(),svm_accuracy.mean(),dt_accuracy.mean()]


# In[62]:


scores_2nd = pd.DataFrame({'Algorithm Name':names,'Score':score})


# In[63]:


axis = sns.barplot(x='Algorithm Name',y='Score',data = scores_2nd)
axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    


# In[64]:


scores_2nd # Cross-validation post HyperParameter Tuning 


# In[65]:


scores # Cross-validation for the Base model


# >Cross-validation post HyperParameter Tuning 

# In[66]:


scores_2nd['Score'].max() 


# >Cross-validation for the Base model

# In[67]:


scores['Score'].max() 


# In[ ]:




