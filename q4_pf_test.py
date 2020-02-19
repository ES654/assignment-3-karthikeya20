import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures


X = np.array([1,2])
poly = PolynomialFeatures(2)
print()
print('input data: ',list(X))
print('max degree: ',2)
print('output: ',poly.transform(X))
