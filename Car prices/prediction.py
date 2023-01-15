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
X = StandardScaler().fit_transform(df)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1)

# Create the model (using Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model
model.score(X_test, y_test)

# Make prediction and the evaluation of the prediction
pred = model.predict(X_test)
score = r2_score(y_test, pred)


