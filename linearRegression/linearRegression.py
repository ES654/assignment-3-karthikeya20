import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as anp
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML, Image
from matplotlib import animation
plt.style.use('seaborn-white')
# Import Autograd modules here

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.n_iter=100
        self.t0s = []
        self.t1s = []
        self.errors = []
        self.surf = False
    
    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.
        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number
        :return None
        '''
        if self.fit_intercept:
            X = pd.concat([pd.Series([1 for i in range(X.shape[0])]),X],axis=1)
        
        self.coef_ = np.zeros(X.shape[1])
        def dot(a,b):
            ans = 0 
            for i in range(len(a)):
                ans += a[i]*b[i]
            return ans
        
        def learning_rate(lr_type,lr,t):
            if lr_type == 'constant':
                return lr
            else:
                return lr/t
            
        def mult(a,k):
            return [k*i for i in a]
            
        n,m = X.shape
        it = 0
        while it<n_iter:
            for i in range(0,n,batch_size):
                gradient = [0 for i in range(m)]
                X_i = X[i:i+batch_size].reset_index(drop=True)
                y_i = y[i:i+batch_size].reset_index(drop=True)
                curr_n = X_i.shape[0]
                for j in range(curr_n):
                    predict = dot(list(X_i.iloc[j]),list(self.coef_))
                    for k in range(m):
                        gradient[k] -= (y_i[j]-predict)*X_i.iloc[j,k]
                self.coef_ = [self.coef_ [mi]- mult(gradient,(1/curr_n)*learning_rate(lr_type,lr,it+1))[mi] for mi in range(len(self.coef_))]
            it+=1
            if it>=n_iter:
                return

    def l_f(self,X,y,theta):
        error = anp.matmul(X,theta)-y
        return (1/X.shape[0])*anp.matmul(error.T,error)


    def fit_vectorised(self, X, y,batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        self.n_iter=n_iter
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number
        :return None
        '''
        if self.fit_intercept:
            X = pd.concat([pd.Series([1 for i in range(X.shape[0])]),X],axis=1)
        self.coef_ = np.zeros((X.shape[1],1))
        def learning_rate(lr_type,lr,t):
            if lr_type == 'constant':
                return lr
            else:
                return lr/t

        n,m = X.shape
        cnt = 0
        self.t0s = []
        self.t1s = []
        while(cnt<n_iter):
            for i in range(0,n,batch_size):
                X_i = X.iloc[i:i+batch_size].reset_index(drop=True)
                y_i = y.iloc[i:i+batch_size].reset_index(drop=True)
                X_i = np.array(X_i)
                y_i = np.array(y_i)
                y_i = np.reshape(y_i, (len(y_i),1))
                self.coef_ -= ((1/X_i.shape[0])*learning_rate(lr_type,lr,cnt+1))*np.matmul(X_i.T,(np.matmul(X_i,self.coef_)-y_i))
                self.t0s.append(self.coef_[0][0])
                self.t1s.append(self.coef_[1][0])
                if m==2:
                    arry=np.array([self.t0s[-1],self.t1s[-1]]).reshape(-1,1)
                    Xlf = np.array(X)
                    ylf = np.array(y).reshape(-1,1)
                    self.errors.append(self.l_f(Xlf,ylf,np.array([self.t0s[-1],self.t1s[-1]]).reshape(-1,1)))
            cnt+=1
            if cnt>=n_iter:
                return


    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        def loss_func(X,y,theta):
            error = anp.matmul(X,theta)-y
            return (1/X.shape[0])*anp.matmul(error.T,error)
        if self.fit_intercept:
            X = pd.concat([pd.Series([1 for i in range(X.shape[0])]),X],axis=1)
        self.coef_ = anp.zeros((X.shape[1],1))
        grad_fun = grad(loss_func,argnum=2)
        def learning_rate(lr_type,lr,t):
            if lr_type == 'constant':
                return lr
            else:
                return lr/t
        n,m = X.shape
        cnt = 0
        while(cnt<n_iter):
            for i in range(0,n,batch_size):
                X_i = X.iloc[i:i+batch_size].reset_index(drop=True)
                y_i = y.iloc[i:i+batch_size].reset_index(drop=True)
                X_i = anp.array(X_i)
                y_i = anp.array(y_i)
                y_i = anp.reshape(y_i, (len(y_i),1))
                # print(np.matmul(X_i.T,(np.matmul(X_i,self.coef_)-y_i)))
                self.coef_ -= 0.5*learning_rate(lr_type,lr,cnt+1)*grad_fun(X_i,y_i,self.coef_)
                # print(self.coef_)
            cnt+=1
            if cnt>=n_iter:
                return

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)),axis=1)
        else:
            X = np.array(X)
        y=np.array(y)
        XtX = np.matmul(X.T,X)
        Xty = np.matmul(X.T,y)
        theta = np.matmul(np.linalg.pinv(XtX),Xty)
        self.coef_ = theta
        return

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if self.fit_intercept:
            X = pd.concat([pd.Series([1 for i in range(X.shape[0])]),X],axis=1)
        y_pred = np.dot(np.array(X),np.array(self.coef_))
        y_pred = y_pred.reshape((y_pred.shape[0],))
        return pd.Series(y_pred)

        

    def plot_surface(self, X, y, error,t_0, t_1,j):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS
        :return matplotlib figure plotting RSS
        """
        #Setup of meshgrid of theta values
        self.surf = True
        T1, T2 = np.meshgrid(np.linspace(0,10,100),np.linspace(0,10,100))
        #Computing the cost function for each theta combination
        zs = np.array(  [self.l_f(np.c_[np.ones(X.shape[0]),X.reshape(-1,1)], y.reshape(-1,1),np.array([t1,t2]).reshape(-1,1))[0][0] for t1, t2 in zip(np.ravel(T1), np.ravel(T2)) ])
        #Reshaping the cost values    
        Z = zs.reshape(T1.shape)
        fig2 = plt.figure(figsize = (16,9))
        ax2 = Axes3D(fig2)
        #Surface plot
        ax2.plot_surface(T1, T2, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
        #ax2.plot(theta_0,theta_1,J_history_reg, marker = '*', color = 'r', alpha = .4, label = 'Gradient descent')

        ax2.set_xlabel('b')
        ax2.set_ylabel('m')
        ax2.set_zlabel('error')
        ax2.view_init(45, -45)

        # Create animation
        line, = ax2.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
        point, = ax2.plot([], [], [], '*', color = 'red')
        display_value = ax2.text(2., 2., 27.5, '', transform=ax2.transAxes)

        def init_2():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            display_value.set_text('')

            return line, point, display_value
        def animate_2(i):
            # Animate line
            line.set_data(t_0[:i], t_1[:i])
            line.set_3d_properties(error[:i])
            
            fig2.suptitle('error:'+str(error[i][0][0]))
            # Animate points
            point.set_data(t_0[i], t_1[i])
            point.set_3d_properties(error[i])
            # Animate display value
            display_value.set_text('Min = ' + str(error[i][0][0]))
            return line, point, display_value
        ax2.legend(loc = 1)
        anim2 = animation.FuncAnimation(fig2, animate_2, init_func=init_2,frames=len(t_0), interval=120, repeat_delay=60, blit=True)
        #plt.show()
        return anim2
        # anim2.save('./gifs/surface_'+str(j)+'.gif', dpi=80,fps=2, writer='imagemagick')
        # plt.close()
        
        
    def format_axes(self,ax):
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('grey')
            ax.spines[spine].set_linewidth(0.5)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_tick_params(direction='out', color='grey')
        return ax

    def plot_line_fit(self, X, Y, t_0s, t_1s,j):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.
        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit
        :return matplotlib figure plotting line fit
        """

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.set_tight_layout(True)

        # Query the figure's on-screen size and DPI. Note that when saving the figure to
        # a file, we need to provide a DPI for that separately.
        print('fig size: {0} DPI, size in inches {1}'.format(
            fig.get_dpi(), fig.get_size_inches()))

        # Plot a scatter that persists (isn't redrawn) and the initial line.

        ax.scatter(X, Y, color='grey', alpha=0.8, s=1)
        # Initial line
        w_prior,b_prior = (0,0)
        line, = ax.plot(X, X*w_prior+b_prior, 'r-', linewidth=1)

        def update(i):
            label = 'Iteration {0}'.format(i)
            line.set_ydata(X*t_1s[i]+t_0s[i])
            ax.set_xlabel(label)
            ax.set_title('m:'+str(t_1s[i])+' '+'b:'+str(t_0s[i]))
            self.format_axes(ax)
            return line, ax

        anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=1)
        return anim

    def plot_contour(self, X, y,error, t_0, t_1,j):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.
        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit
        :return matplotlib figure plotting the contour
        """
        T1, T2 = np.meshgrid(np.linspace(-50,50,100),np.linspace(-5,15,100))
        #Computing the cost function for each theta combination
        zs = np.array(  [self.l_f(np.c_[np.ones(X.shape[0]),X.reshape(-1,1)], y.reshape(-1,1),np.array([t1,t2]).reshape(-1,1))[0][0] for t1, t2 in zip(np.ravel(T1), np.ravel(T2)) ])
        #Reshaping the cost values    
        Z = zs.reshape(T1.shape)
        
        fig1, ax1 = plt.subplots(figsize = (16,9))
        ax1.contour(T1, T2, Z, 100, cmap = 'jet')
        ax1.set_xlabel('b')
        ax1.set_ylabel('m(slope)')


        # Create animation
        line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
        point, = ax1.plot([], [], '*', color = 'red', markersize = 4)
        value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

        def init_1():
            line.set_data([], [])
            point.set_data([], [])
            value_display.set_text('')

            return line, point, value_display

        def animate_1(i):
            # Animate line
            line.set_data(t_0[:i], t_1[:i])
            fig1.suptitle('error:'+str(error[i][0][0]))

            # Animate points
            point.set_data(t_0[i], t_1[i])

            # Animate value display
            value_display.set_text('Min = ' + str(error[i][0][0]))

            return line, point, value_display

        ax1.legend(loc = 1)

        anim1 = animation.FuncAnimation(fig1, animate_1, init_func=init_1,
                                    frames=len(t_0), interval=100, 
                                    repeat_delay=60, blit=True)
        return anim1
