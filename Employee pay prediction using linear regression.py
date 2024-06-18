#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression Exercise
# 
# Exercise: Predict the pay of an employee based on their years of experience and designation.
# 
# Designation Levels in the organization	
# 1	Executive
# 2	Manager
# 3	Senior manager
# 4	Director
# 

# In[1]:


# To enable plotting graphs in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Numerical libraries
import numpy as np   

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression

# to handle data in form of rows and columns 
import pandas as pd    

# importing ploting libraries
import matplotlib.pyplot as plt   
import matplotlib.style
plt.style.use('classic')

#importing seaborn for statistical plots
import seaborn as sns


# In[5]:


# reading the CSV file into pandas dataframe
df = pd.read_csv("EmployeesData.csv")


# In[6]:


# Check top few records to get a feel of the data structure
df.head(50)


# In[9]:


#Lets analysze the distribution of the data
df.describe().transpose()


# In[38]:


#understanding the datatypes
df.dtypes


# In[39]:


#checking for any null values
df[df.isnull().any(axis=1)]


# In[40]:


#No null values in this case but if you find null values you can either drop the values or try replacing nan with median
#df = df.apply(lambda x: x.fillna(x.median()),axis=0)


# In[41]:


# Getting The five-number summary
df.describe()


# In[42]:


#Median salary for each designation
df.groupby('Designation')['Pay'].median()


# In[43]:


#viewing the data in histograms
df.hist(figsize = (20,30))


# In[44]:


# Let us do a correlation analysis among the different dimensions and also each dimension with the dependent dimension
# This is done using scatter matrix function which creates a dashboard reflecting useful information about the dimensions
sns.pairplot(df, diag_kind='kde')   # to plot density curve instead of histogram


# In[45]:


fig = sns.boxplot(x='Designation', y="Pay", data=df)


# In[46]:


fig = sns.boxplot(x='Years of experience', y="Pay", data=df)


# In[47]:


df.boxplot(figsize=(15, 10))


# ### Correlations
# 
# In statistics, Spearman's rank correlation coefficient or Spearman's rho, named after Charles Spearman and often denoted by the Greek letter $\rho$ (rho) or as $r_{s}$, is a nonparametric measure of rank correlation (statistical dependence between the rankings of two variables). It assesses how well the relationship between two variables can be described using a monotonic function.
# 
# The Spearman correlation between two variables is equal to the Pearson correlation between the rank values of those two variables; while Pearson's correlation assesses linear relationships, Spearman's correlation assesses monotonic relationships (whether linear or not). If there are no repeated data values, a perfect Spearman correlation of +1 or −1 occurs when each of the variables is a perfect monotone function of the other.

# In[48]:


import matplotlib.pylab as plt
corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, linewidths=1)


# In order to run our models on the data, I had to transform many of the variables. The following pre-processing steps can be taken:
# 
# - Removing outliers: In case the classic Tukey method of taking 1.5 * IQR to remove outliers removes too much data then remove values that are outside of 3 * IQR instead.
# 
# - Filling NaN values: As there are no null values found in our dataset, so need to handle it.
# 
# - Create dummy variables for the categorical variables (not needed)
# 
# - Split the data into a training set and a test set
# 
# - Scaled the data (in case of non linear relashionship)

# In[49]:


# Copy all the predictor variables into X dataframe
X = df.drop('Pay', axis=1)

# Copy the 'Pay' column alone into the y dataframe. This is the dependent variable
y = mpg_df[['Pay']]


# In[50]:


#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split


# In[51]:


# Split X and y into training and test set in 75:25 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)


# In[52]:


# invoke the LinearRegression function and find the bestfit model on training data

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# In[53]:


# Let us explore the coefficients for each of the independent attributes

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))


# In[54]:


# Let us check the intercept for the model

intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


# In[55]:


regression_model.score(X_train, y_train)


# In[56]:


# So the model explains 93.5% of the variability in Y using X

