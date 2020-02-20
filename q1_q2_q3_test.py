
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from copy import deepcopy

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
niter = 100
print()
for j in ['constant','inverse']:
    print(j,':')
    print()
    print('Non Vectorised:')
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_non_vectorised(X, y, 1 , n_iter=niter,lr_type = j) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()
    print('Vectorised')
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_vectorised(X, y, 1,n_iter=niter,lr_type = j) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()
    print('Autograd')
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_autograd(X, y, 1,n_iter=niter,lr_type = j) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)

        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
    print()
    print('Fit Normal')
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)

        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))



