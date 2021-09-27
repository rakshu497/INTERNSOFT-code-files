# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:16:18 2021

@author: Rakshithkumar
"""

#Linear regression: is a preditctive modren technique which gives out the relationship between dependent variables and independent variables
# regression on continues data
#why do we use regression


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# READING THE DATA FROM YOUR FILES
data = pd.read_csv('advertising.csv')
print(data.head())

# TO VISUALIZE DATA 
fig, axs = plt.subplots(1,3,sharey=True)
data.plot(kind = 'scatter', x = 'TV', y = 'Sales', ax= axs[0],figsize=(14,7))
data.plot(kind = 'scatter', x = 'Radio', y = 'Sales', ax= axs[1])
data.plot(kind = 'scatter', x = 'Newspaper', y = 'Sales', ax= axs[2])

# CREATING  X & Y FOR LINEAR REGRESSION
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

#IMPORTING LINEAR REGRESSION ALGORITHM
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)
print(lr.fit(X,y))

print(lr.intercept_)
print(lr.coef_)

#y = a+bx

result = 6.9748214882298925+0.05546477*50
print(result)


#CREATE A DATA FRAME WITH MIN AND MAX VALUES OF THE TABLE

X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
print(X_new.head())


preds = lr.predict(X_new)
print(preds)

data.plot(kind = 'scatter', x='TV', y='Sales')
# least squared line in linear regression
#X_new is min & max value
# to find the best fit line
plt.plot(X_new,preds,c='red',linewidth=3)



import statsmodels.formula.api as smf
lr = smf.ols(formula ='Sales ~ TV',data=data).fit()
lr.conf_int()
print(lr.conf_int())


# FINDING THE PROBABILITY VALUES

print(lr.pvalues)


# FINDING THE R-SQUARED VALUES

print(lr.rsquared)



# MULTI LINEAR  REGRESSION 
feature_cols = ['TV', 'Radio','Newspaper']
X = data[feature_cols]
y =data.Sales

lr = LinearRegression()
lr.fit(X,y)



print(lr.intercept_)
print(lr.coef_)

lr = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data=data).fit()
print(lr.conf_int())

print(lr.summary())

lr = smf.ols(formula = 'Sales ~ TV+Radio',data=data).fit()
print(lr.conf_int())

print(lr.summary())






