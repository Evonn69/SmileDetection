import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Obtain the benelux and store it in a variable and return the top 5 rows to check whether its the right data
dataset = pd.read_csv("C:/Users/reibo/OneDrive/Dokumenti/dokumenti za faks/year 3/machine  learning/Benelux suicides/master.csv")

# Obtain the list of all countries filtered by the name
unique_country = dataset['country'].unique()

# Plot the count of data (rows) per each country
#plt.figure(figsize=(10,25))
#sns.countplot(y='country', data=benelux, alpha=0.6)
#plt.title('Data by country')
#plt.show()

# Gather the values for the Benelux countries
df = dataset
benelux = df[(df['country'].isin(['Belgium', 'Luxembourg', 'Netherlands']))]

# Store a range of ages corresponding to the age groups in variables so the lineplot is easier to code later
age_5 = benelux.loc[benelux.loc[:, 'age'] == '5-14 years',:]
age_15 = benelux.loc[benelux.loc[:, 'age'] == '15-24 years',:]
age_25 = benelux.loc[benelux.loc[:, 'age'] == '25-34 years',:]
age_35 = benelux.loc[benelux.loc[:, 'age'] == '35-54 years',:]
age_55 = benelux.loc[benelux.loc[:, 'age'] == '55-74 years',:]
age_75 = benelux.loc[benelux.loc[:, 'age'] == '75+ years',:]

# Store the male and female values over these years
male_population = benelux.loc[benelux.loc[:, 'sex'] == 'male',:]
female_population = benelux.loc[benelux.loc[:, 'sex'] == 'female',:]

# Store the countries' values throughout these years
belgium = benelux.loc[benelux.loc[:, 'country'] == 'Belgium',:]
netherlands = benelux.loc[benelux.loc[:, 'country'] == 'Netherlands',:]
luxembourg = benelux.loc[benelux.loc[:, 'country'] == 'Luxembourg',:]

# Set the figure size
plt.figure(figsize=(16,7))

# Plot the lines
age_5_line = sns.lineplot(x = 'year', y = 'suicides_no', data = age_5)
age_15_line = sns.lineplot(x = 'year', y = 'suicides_no', data = age_15)
age_25_line = sns.lineplot(x = 'year', y = 'suicides_no', data = age_25)
age_35_line = sns.lineplot(x = 'year', y = 'suicides_no', data = age_35)
age_55_line = sns.lineplot(x = 'year', y = 'suicides_no', data = age_55)
age_75_line = sns.lineplot(x = 'year', y = 'suicides_no', data = age_75)

# Plot the lines
#lp_male = sns.lineplot(x = 'year' , y = 'suicides_no' , data = male_population)
#lp_female = sns.lineplot(x = 'year' , y = 'suicides_no' , data = female_population)

# Plot the lines
#lp_belgium = sns.lineplot(x = 'year' , y = 'suicides_no' , data = belgium)
#lp_luxembourg = sns.lineplot(x = 'year' , y = 'suicides_no' , data = luxembourg)
#lp_netherlands = sns.lineplot(x = 'year' , y = 'suicides_no' , data = netherlands)

# Create the legend for lineplotting
#sex = sns.countplot(x='sex',data = benelux)
#plt.title('Count by gender')
#cor = sns.heatmap(benelux.corr(), annot = True)
#plt.title('Data correlation')
#bar_age = sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age', data = benelux)
#plt.title('Suicides by age')
#bar_gen = sns.barplot(x = 'sex', y = 'suicides_no', hue = 'generation', data = benelux)
#plt.title('Suicides by generation')
#cat_accord_year = sns.catplot(x = 'sex', y = 'suicides_no', hue = 'age', col = 'year', data = benelux, kind = 'bar', col_wrap = 5)
leg = plt.legend(['5-14 years', '5-14 years', '15-24 years', '15-24 years', '25-34 years', '25-34 years', '35-54 years', '35-54 years', '55-74 years', '55-74 years', '75+ years', '75+ years'])
#leg = plt.legend(['Males', 'Males',' Females', 'Females'])
#leg = plt.legend(['Belgium', 'Belgium', 'Luxembourg', 'Luxembourg', 'Netherlands', 'Netherlands'])

plt.title('Suicides over the years within age groups')
#plt.title('Suicides over the years by males and females')
#plt.title('Suicides over the years distributed by the Benelux countries')
plt.show()
