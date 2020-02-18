
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
niter = 1
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


X = np.array([i*np.pi/180 for i in range(60,300,4)])  #Setting seed for reproducibility
y = 4*(X) + 7 + np.random.normal(0,3,len(X))

for j in ['constant','inverse']:
    LR = LinearRegression(fit_intercept=True)
    LR.fit_vectorised(pd.DataFrame(np.array([list(X)]).T), pd.Series(y), 100,n_iter=10,lr_type = j,lr=0.02) # here you can use fit_non_vectorised / fit_autograd methods
    t_0s=LR.t0s
    t_1s=LR.t1s
    n_iter = LR.n_iter
    error = LR.errors
    LR.plot_line_fit(np.array(X), np.array(y), t_0s, t_1s,j)
    LR.plot_surface(np.array(X), np.array(y), error,t_0s, t_1s,j)
    LR.plot_contour(np.array(X), np.array(y), error,t_0s, t_1s,j)


