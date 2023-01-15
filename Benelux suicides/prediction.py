import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Obtain the benelux and store it in a variable
df = pd.read_csv("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/machine  learning/Benelux suicides/master.csv")

# Gather the values for the Netherlands
netherlands = df[(df['country'].isin(['Netherlands']))]

# Set the figure size
plt.figure(figsize=(16,7))
cor = sns.heatmap(netherlands.corr(), annot = True)
plt.title('Data correlation')
plt.show()

# Select the features used for predicting the target
features = ['year', 'gdp_per_capita ($)']
target = 'suicides_no'

X = df[features]
y = df[target]

# Split the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

# Test the model 
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)
