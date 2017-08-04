# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("2015.csv")
df.columns

#Show the correlation between features
plt.figure(figsize=(10,8))
corr = df.drop(['Country','Region','Happiness Rank','Standard Error'],axis = 1).corr()
sns.heatmap(corr, cbar = True, square = True, annot=True, linewidths = .5, fmt='.2f',annot_kws={'size': 15}) 
sns.plt.title('Heatmap of Correlation Matrix')
plt.show()