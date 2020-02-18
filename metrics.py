import numpy as np 
import pandas as pd

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    pass
    size=len(y)
    correct=0
    for i in range(size):
        if y[i]==y_hat[i]:
            correct+=1
    return correct/size

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    size=len(y)
    true_positive=0
    false_positive=0
    for i in range(size):
        if y_hat[i]==cls:
            if y[i]==cls:
                true_positive+=1
            else:
                false_positive+=1
    if (true_positive+false_positive) != 0 :
        return true_positive/(true_positive+false_positive)        
    return 0

def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    size=len(y)
    true_positive=0
    false_negative=0
    for i in range(size):
        if y[i]==cls:
            if y_hat[i]==cls:
                true_positive+=1
            else:
                false_negative+=1
    return true_positive/(true_positive+false_negative)        

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)
    size=y.size
    y=y.reset_index(drop=True)
    y_hat=y_hat.reset_index(drop=True)
    SE=0
    for i in range(size):
        SE+=(y_hat[i]-y[i])**2
    MSE=SE/size
    RMSE=MSE**(0.5)
    return RMSE

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    size=y.size
    y=y.reset_index(drop=True)
    y_hat=y_hat.reset_index(drop=True)
    AE=0
    for i in range(size):
        AE+=abs(y_hat[i]-y[i])
    MAE=AE/size  
    return MAE
