# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Obtain the dfset and store it in a variable
df = pd.read_csv("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/machine  learning/Car prices/car data.csv")

# Creating dummies for the above variables
Fuel_type = pd.get_dummies(df['Fuel_Type'], drop_first = True)
Seller_Type = pd.get_dummies(df['Seller_Type'], drop_first = True)
Transmission = pd.get_dummies(df['Transmission'], drop_first = True)

# Drop the dummy variables from the data frame and combine them into a single one
df = df.drop(['Fuel_Type', 'Seller_Type', 'Transmission', 'Car_Name'], axis=1)
df = pd.concat([df,Fuel_type, Seller_Type, Transmission], axis=1)

# Define the target variable
Y = df['Selling_Price']
X = df.drop(['Selling_Price'], axis=1)
#X = StandardScaler().fit_transform(df)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1)

# Create the model (using Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
#print(model.score(X_test, y_test))

# Compare the prediction with the actual values and show the error
pred = model.predict(X_test)
pred_overview = pd.DataFrame()
pred_overview["truth"] = y_test
pred_overview["pred"] = pred
pred_overview["error"] = pred_overview["truth"] - pred_overview["pred"]
pred_overview["error"] = abs(pred_overview["error"].astype(int))
pred_overview = pred_overview.reset_index(drop=  True)
#print(pred_overview)

score = r2_score(y_test, pred)
#print(score)

plot = sns.regplot(y=y_test.values.flatten(), x=pred.flatten(), line_kws={"color": "g"})
plot.set_xlabel("predicted price")
plot.set_ylabel("actual price")
plt.show()

