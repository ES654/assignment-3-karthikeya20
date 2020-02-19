import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from copy import deepcopy
x = np.array([[i*np.pi/180 for i in range(60,300,4)]]).T
np.random.seed(10)  #Setting seed for reproducibility
y1 = 4*(x.T[0]) + 7 + np.random.normal(0,3,len(x.T[0]))
N = [10,20,30,40,50,60]
for n in N:
    y = pd.Series(y1[:n])
    max_degree = 10
    degrees = [i+1 for i in range(max_degree)]
    max_thetas = []
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        x_new = poly.transform(x[:n])
        X = pd.DataFrame(x_new)
        LR = deepcopy(LinearRegression(fit_intercept=False))
        LR.fit_vectorised(X, y, n, n_iter=5, lr =0.0001) # here you can use fit_non_vectorised / fit_autograd methods
        thetas = LR.coef_
        max_theta = np.linalg.norm(thetas,ord=np.inf)
        max_thetas.append(max_theta)
    plt.plot(degrees,max_thetas,label = 'N='+str(n))
    plt.yscale('log')
    plt.title(r'Plot of |$\theta$| vs Degree for polynomial fit (log scale)')
    plt.xlabel('Degree')
    plt.ylabel(r'|$\theta$|')
    plt.legend()
plt.savefig('./gifs/q6_plot.png')
plt.show()