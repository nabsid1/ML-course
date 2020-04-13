'''
#Nabeel Siddiqui 
#Assignment no.3

Question no.1
Take 50 startups of any two countries and find out which country is going to provide best profit in future.

'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#Florida
#finding investment
dataset['Total Spend']= dataset['R&D Spend'] + dataset['Administration'] + dataset['Marketing Spend'  ]
Invested = dataset.loc[(dataset['State'] == 'Florida'), ['Total Spend']] 
Profit = dataset.loc[(dataset['State'] == 'Florida'), ['Profit']]       

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Invested,Profit)

# Visualising the Decision Tree Regression
plt.scatter(Invested, Profit, color = 'orange')
plt.plot(Invested, regressor.predict(Invested), color = 'black')
plt.title('Florida - Total Spend vs Profit (Decision Tree Regression)')
plt.xlabel('Invested')
plt.ylabel('Profit')
plt.show()
print ("Florida = " ,regressor.predict([[220000]]))

#California
#finding investment
dataset['Total Spend']= dataset['R&D Spend'] + dataset['Administration'] + dataset['Marketing Spend'  ]         # x-axis
Invested = dataset.loc[(dataset['State'] == 'California'), ['Total Spend']] 
Profit = dataset.loc[(dataset['State'] == 'California'), ['Profit']]       

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(Invested, Profit)

# Visualising the Decision Tree Regression
plt.scatter(Invested, Profit, color = 'orange')
plt.plot(Invested, regressor.predict(Invested), color = 'black')
plt.title('California - Total Spend vs Profit (Decision Tree Regression)' )
plt.xlabel('Invested')
plt.ylabel('Profit')
plt.show()
print ("California = " ,regressor.predict([[220000]]))

print('As Shown from the plots Florida will provide more profits as compaired to California ')