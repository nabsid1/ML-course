'''
#Nabeel Siddiqui 
#Assignment no. 3



Question no. 5
Data of monthly experience and income distribution of different employs is given. Perform regression.

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('monthlyexp vs incom.csv')

#FOR GCAG
A = dataset.loc[: , ['MonthsExperience']]
B = dataset.loc[: , ['Income']]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(A, B)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 2)
A_poly = polyreg.fit_transform(A)
polyreg.fit(A_poly, B)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, B)

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'orange')
plt.plot(A, lin_reg_2.predict(polyreg.fit_transform(A)), color = 'Black')
plt.title('monthly experience vs income(Poly-Regression)')
plt.xlabel('Month')
plt.ylabel('INCOME')
plt.show()

# Predicting with Polynomial Regression
print('future analysis')
print("Prediction of income from 20 months of Experience" , lin_reg_2.predict(polyreg.fit_transform([[20]])))
print("Prediction of income from 22 months of Experience" , lin_reg_2.predict(polyreg.fit_transform([[22]])))
print("Prediction of income from 25 months of Experience" , lin_reg_2.predict(polyreg.fit_transform([[25]])))
