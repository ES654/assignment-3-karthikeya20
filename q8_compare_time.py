import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time
from metrics import *
from tqdm import tqdm
np.random.seed(42)
# time complexity analysis
fvo=[]
fnvo=[]
fno=[]
po=[]
fve=[]
fnve=[]
fne=[]
pe=[]
def fit_t(n,m):
    return n*m**2+m**3

niter = 5
for N in tqdm(range(1,1000,200)):
    fvol=[]
    fnvol=[]
    fnol=[]
    pol=[]
    fvel=[]
    fnvel=[]
    fnel=[]
    pel=[]    
    for P in range(1,20,4):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
        for fit_intercept in [True]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            fit_time = time.time()
            LR.fit_non_vectorised(X, y, batch_size =N,n_iter=niter,lr_type = 'constant') # here you can use fit_non_vectorised / fit_autograd methods
            fit_time = time.time() - fit_time
            fnvol.append(fit_time)
            LR = LinearRegression(fit_intercept=fit_intercept)
            fit_time = time.time()
            LR.fit_vectorised(X, y, batch_size =N, n_iter=niter,lr_type = 'constant') # here you can use fit_non_vectorised / fit_autograd methods
            fit_time = time.time() - fit_time
            fvol.append(fit_time)
            LR = LinearRegression(fit_intercept=fit_intercept)
            fit_time = time.time()
            LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
            fit_time = time.time() - fit_time
            fnol.append(fit_time)
            pred_time = time.time()
            LR.predict(X) # here you can use fit_non_vectorised / fit_autograd methods
            pred_time = time.time() - pred_time
            pol.append(pred_time)
            fnel.append(fit_t(N,P)/(10**8))
            pel.append((N*P)/10**8)
            fvel.append((N*P*5)/10**8)
            fnvel.append((N*P*5)/10**8)
    fvo.append(fvol)
    fnvo.append(fnvol)
    fno.append(fnol)               
    po.append(pol)
    fve.append(fvel)
    fnve.append(fnvel)
    fne.append(fnel)               
    pe.append(pel)
a,b,c,d,e,f=[1,1000,200,1,20,4]

def help_plot(am,grid,row,col,name):
    n = [i for i in range(a,b,c)]
    m = [i for i in range(d,e,f)]
    ax = am[row][col]
    im = ax.imshow(grid,cmap='viridis')
    ax.set_xticks(np.arange(len(m)))
    ax.set_yticks(np.arange(len(n)))
    ax.set_xticklabels(m)
    ax.set_yticklabels(n)
    ax.set_xlabel('M')
    ax.set_ylabel('N')
    for i in range(len(n)):
        for j in range(len(m)):
            # print(i,j)
            # print(len(grid),len(grid[0]))
            # print(grid)
            text = ax.text(j, i, round(grid[i][j],3),ha="center", va="center", color="w")
    ax.set_title(name )
    plt.colorbar(im,ax=ax)
    

def plot():
    fig, am = plt.subplots(2,4)
    fig.suptitle('Time taken to fit and predict in various methods')
    help_plot(am,fvo,0,0,'fit vectoried observed')
    help_plot(am,fnvo,0,1,'fit non vectoried observed')
    help_plot(am,fno,0,2,'fit normal observed')
    help_plot(am,po,0,3,'predict observed')
    help_plot(am,fve,1,0,'fit vectoried estimated')
    help_plot(am,fnve,1,1,'fit non vectoried estimated')
    help_plot(am,fne,1,2,'fit normal estimated')
    help_plot(am,pe,1,3,'predict estimated')
    plt.show()
    fig.savefig('./gifs/timing.png')
plot()