'''
#Nabeel Siddiqui 
#Assignment no.3

Question no.3
Data of global production of CO2 of a place is given between 1970s to 2010. Predict the CO2 production for the years 2011, 2012 and 2013 using the old data set.

'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('global_co2.csv')

#FOR GCAG
A = dataset.loc[: , ['Year']]
B = dataset.loc[: , ['Total']]

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
plt.title('Year VS Total CO2(Poly-Regression)')
plt.xlabel('Year')
plt.ylabel('Total')
plt.show()

# Predicting with Polynomial Regression
print("Prediction of CO2 in 2011: " , lin_reg_2.predict(polyreg.fit_transform([[2011]])))
print("Prediction of CO2 in 2012: " , lin_reg_2.predict(polyreg.fit_transform([[2012]])))
print("Prediction of CO2 in 2013: " , lin_reg_2.predict(polyreg.fit_transform([[2013]])))
