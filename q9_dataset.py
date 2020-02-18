import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from copy import deepcopy

np.random.seed(42)

N = 30
P = 5
X1 = pd.DataFrame(np.random.randn(N, P))
X = pd.concat([X1,2*X1[3],5*X1[4]],axis=1)
y = pd.Series(np.random.randn(N))
niter = 100
print('with multicollinearity')
print()
for j in ['constant','inverse']:
    print('learning rate',j,':')
    print()
    print('Vectorised:')
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_vectorised(X, y, 30 , n_iter=niter,lr_type = j) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()
print()
print()
print('without multicollinearity')
print()
for j in ['constant','inverse']:
    print('learning rate',j,':')
    print()
    print('Vectorised:')
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_vectorised(X1, y, 30 , n_iter=niter,lr_type = j) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X1)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()
