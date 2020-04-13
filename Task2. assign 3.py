'''
#Nabeel Siddiqui 
#Assignment no.3

Question no.2
Annual temperature between two industries is given. Predict the temperature in 2016 and 2017 using the past data of both country.

'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('annual_temp.csv')

#FOR GCAG
A = dataset.loc[(dataset.Source == 'GCAG'), ['Year']]
B = dataset.loc[(dataset.Source == 'GCAG'), ['Mean']]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(A, B)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 8)
A_poly = polyreg.fit_transform(A)
polyreg.fit(A_poly, B)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, B)

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'orange')
plt.plot(A, lin_reg_2.predict(polyreg.fit_transform(A)), color = 'Black')
plt.title('Year VS Annual Temp For GCAG(Poly-Regression)')
plt.xlabel('Year')
plt.ylabel('Temp')
plt.show()

# Predicting with Polynomial Regression
print("Prediction of GCAG in 2016: " , lin_reg_2.predict(polyreg.fit_transform([[2016]])))
print("Prediction of GCAG in 2017: " , lin_reg_2.predict(polyreg.fit_transform([[2017]])))

#FOR GISTEMP
A1 = dataset.loc[(dataset.Source == 'GISTEMP'), ['Year']]
B1 = dataset.loc[(dataset.Source == 'GISTEMP'), ['Mean']]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(A1, B1)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 8)
A1_poly = polyreg.fit_transform(A1)
polyreg.fit(A1_poly, B1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A1_poly, B1)

# Visualising the Polynomial Regression results
plt.scatter(A1, B1, color = 'orange')
plt.plot(A1, lin_reg_2.predict(polyreg.fit_transform(A1)), color = 'Black')
plt.title('Years VS Annual Temp For GISTEMP(Poly-Regression)')
plt.xlabel('Year')
plt.ylabel('Temp')
plt.show()

# Predicting with Polynomial Regression
print("Prediction of GISTEMP in 2016: " , lin_reg_2.predict(polyreg.fit_transform([[2016]])))
print("Prediction of GISTEMP in 2017: " , lin_reg_2.predict(polyreg.fit_transform([[2017]])))