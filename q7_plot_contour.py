import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
np.random.seed(42)


X = np.array([i*np.pi/180 for i in range(60,300,4)])  #Setting seed for reproducibility
y = 4*(X) + 7 + np.random.normal(0,3,len(X))

print('surface:')
print()
for j in ['constant','inverse']:
    LR = LinearRegression(fit_intercept=True)
    LR.fit_vectorised(pd.DataFrame(np.array([list(X)]).T), pd.Series(y), 100,n_iter=10,lr_type = j,lr=0.02) # here you can use fit_non_vectorised / fit_autograd methods
    t_0s=LR.t0s
    t_1s=LR.t1s
    n_iter = LR.n_iter
    error = LR.errors
    #LR.plot_line_fit(np.array(X), np.array(y), t_0s, t_1s,j)
    anim2 = LR.plot_surface(np.array(X), np.array(y), error,t_0s, t_1s,j)
    #LR.plot_contour(np.array(X), np.array(y), error,t_0s, t_1s,j)
    anim2.save('./gifs/plotsurface_'+str(j)+'.gif', dpi=80,fps=2, writer='imagemagick')
    print(j,':')
    plt.show()
    plt.pause(3)
    plt.close()
    
print('line_prior:')
print()
for j in ['constant','inverse']:
    LR = LinearRegression(fit_intercept=True)
    LR.fit_vectorised(pd.DataFrame(np.array([list(X)]).T), pd.Series(y), 100,n_iter=10,lr_type = j,lr=0.02) # here you can use fit_non_vectorised / fit_autograd methods
    t_0s=LR.t0s
    t_1s=LR.t1s
    n_iter = LR.n_iter
    error = LR.errors
    #LR.plot_line_fit(np.array(X), np.array(y), t_0s, t_1s,j)
    anim = LR.plot_line_fit(np.array(X), np.array(y), t_0s, t_1s,j)
    #LR.plot_contour(np.array(X), np.array(y), error,t_0s, t_1s,j)
    anim.save('./gifs/plotline_prior'+str(j)+'.gif', dpi=80,fps=2, writer='imagemagick')
    print(j,':')
    plt.show()
    plt.pause(3)
    plt.close()

print('contour:')
print()
for j in ['constant','inverse']:
    LR = LinearRegression(fit_intercept=True)
    LR.fit_vectorised(pd.DataFrame(np.array([list(X)]).T), pd.Series(y), 100,n_iter=10,lr_type = j,lr=0.02) # here you can use fit_non_vectorised / fit_autograd methods
    t_0s=LR.t0s
    t_1s=LR.t1s
    n_iter = LR.n_iter
    error = LR.errors
    #LR.plot_line_fit(np.array(X), np.array(y), t_0s, t_1s,j)
    anim1 = LR.plot_contour(np.array(X), np.array(y), error,t_0s, t_1s,j)
    #LR.plot_contour(np.array(X), np.array(y), error,t_0s, t_1s,j)
    anim1.save('./gifs/plotcontour_'+str(j)+'.gif', dpi=80,fps=2, writer='imagemagick')
    print(j,':')
    plt.show()
    plt.pause(3)
    plt.close()

