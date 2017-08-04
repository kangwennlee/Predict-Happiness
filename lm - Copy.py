# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("2015.csv")
df.columns

df2 = pd.read_csv("2016.csv")
df2.columns

#Setting up linear model to predict happiness
X = df.drop(['Standard Error','Happiness Score', 'Happiness Rank', 'Country', 'Region'], axis=1)
y = df['Happiness Score']
X.columns

X2 = df2.drop(['Happiness Score', 'Happiness Rank', 'Country', 'Region','Lower Confidence Interval', 'Upper Confidence Interval'], axis=1)
y2 = df2['Happiness Score']
X2.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#print('Standardized features\n')
#print(str(X_train[:4]))

scaler.fit(X2)
X2 = scaler.transform(X2)

from sklearn import linear_model
lm = linear_model.LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)

result_lm = pd.DataFrame({
    'Actual':y_test,
    'Predict':y_pred
})
result_lm['Diff'] = y_test - y_pred
result_lm.head()

y_pred2 = lm.predict(X2)

result_lm2 = pd.DataFrame({
    'Actual':y2,
    'Predict':y_pred2
})
result_lm2['Diff'] = y2 - y_pred2
result_lm2.head()

sns.regplot(x='Actual',y='Predict',data=result_lm2)

from sklearn import metrics
# The coefficients
print('Coefficients: \n', lm.coef_)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lm.score(X_test, y_test))

sns.regplot(x='Actual',y='Predict',data=result_lm)
