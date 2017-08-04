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

#Setting up linear model to predict happiness
X = df.drop(['Happiness Score', 'Happiness Rank', 'Country', 'Region'], axis=1)
y = df['Happiness Score']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#print('Standardized features\n')
#print(str(X_train[:4]))

from sklearn import linear_model
#clf = linear_model.BayesianRidge()
clf = linear_model.HuberRegressor()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

result_clf = pd.DataFrame({
        'Actual':y_test,
        'Predict':y_pred
})
result_clf['Diff'] = y_test - y_pred
result_clf.head()

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sns.regplot(x='Actual',y='Predict',data=result_clf)