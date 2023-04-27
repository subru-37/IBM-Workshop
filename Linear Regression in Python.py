#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression in Machine Learning

# In[1]:


import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# In[2]:


mydata=mypd.read_csv("C:/Users/DELL/Desktop/Dataset/Mult_Reg_Yield.csv")
mydata


# In[3]:


#To check missing values
mydata.describe()


# In[4]:


mydata.info()


# In[5]:


#Separating x's and y's
x = mydata.iloc[:, 0:2]
x


# In[6]:


#Separating x's and y's
y = mydata.Yield
y


# In[7]:


#Correlation Analysis
# Scatter plot
mysb.pairplot(mydata)
myplot.show()


# In[10]:


#Regression Modeling
# fitting the model
mymodel = LinearRegression()
mymodel


# In[11]:


mymodel = mymodel.fit(x,y)
mymodel


# In[12]:


mymodel.coef_


# In[13]:


mymodel.intercept_


# In[15]:


# Model accuracy-R Square value
rsq = mymodel.score(x,y)
rsq


# In[16]:


round(rsq*100,2)


# In[17]:


pred = mymodel.predict(x)
pred


# In[18]:


mse = mean_squared_error(y, pred)
mse


# In[20]:


import math as mymath
rmse = mymath.sqrt(mse)
rmse


# In[21]:


# Residual Analysis
res = y-pred
res


# In[23]:


pred=mypd.DataFrame(pred,columns=['Predicted'])
pred


# In[24]:


myresult=mydata.join(pred)
myresult


# In[25]:


from scipy import stats


# In[20]:


#Residual Analysis – Actual Vs Predicted Plot
myplot.scatter(y, pred)
myplot.title("Actual vs Predicted Plot")
myplot.xlabel("Actual Yield")
myplot.ylabel("Predicted yield")
myplot.show()


# In[21]:


#Residual Analysis – Predicted Vs Residuals Plot
myplot.scatter(pred, res)
myplot.title("Predicted vs Residuals Plot")
myplot.xlabel("Predicted")
myplot.ylabel("Residuals")
myplot.grid()
myplot.show()


# In[23]:


#Residual Analysis: Normality test
norm_test = stats.normaltest(res)
w = norm_test[0]
p_value = norm_test[1]


# In[24]:


stats.probplot(res, plot= myplot)
myplot.grid()
myplot.show()


# In[25]:


stats.normaltest(res)
#gives p value>0.05. So residuals are normally distributed


# In[26]:


# Cross Validation-Model Generalizability check(compare mse and rmse for original one and those obtained after cross validation)
myscore = cross_val_score(mymodel,x, y, scoring='neg_mean_squared_error', cv = 4)
myscore


# In[27]:


cv_mse=-1*myscore.mean()
cv_mse


# In[21]:


rmse = mymath.sqrt(cv_mse)
rmse


# In[22]:


#Cross validation residual sum of squares
cv_rss=cv_mse*16
cv_rss


# In[23]:


#Total Sum of Squares
total_ss=y.var()*(16-1)
total_ss


# In[24]:


cv_rsq=1-(cv_rss/total_ss)
cv_rsq


# In[ ]:




