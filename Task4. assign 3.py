'''
#Nabeel Siddiqui 
#Assignment no. 3


Question no. 4
Housing price according to the ID is assigned to every-house. Perform future analysis where when ID is inserted the housing price is displayed.

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('housing price.csv')

#FOR GCAG
A = dataset.loc[: , ['Id']]
B = dataset.loc[: , ['SalePrice']]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(A, B)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 10)
A_poly = polyreg.fit_transform(A)
polyreg.fit(A_poly, B)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, B)

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'orange')
plt.plot(A, lin_reg_2.predict(polyreg.fit_transform(A)), color = 'Black')
plt.title('ID Vs SalePrice (Poly-Regression)')
plt.xlabel('ID#')
plt.ylabel('Sale Price')
plt.show()

# Predicting with Polynomial Regression
print('future analysis')
print("Prediction of Sales at ID# 3000" , lin_reg_2.predict(polyreg.fit_transform([[3000]])))
print("Prediction of Sales at ID# 1000" , lin_reg_2.predict(polyreg.fit_transform([[1000]])))
print("Prediction of Sales at ID# 2000" , lin_reg_2.predict(polyreg.fit_transform([[2000]])))
