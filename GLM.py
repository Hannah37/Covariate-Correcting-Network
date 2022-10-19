import pandas as pd
import os
from scipy.stats import f
from scipy import stats
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_XYZ(df):
    cols = ['ddtidp_' + str(i) for i in range(605, 605+148)]
    Y = df[cols] 
    Z = df[['age_x', 'age_y', 'gender', 'GE', 'Philips', 'SIEMENS', 'White', 'Black', 'Others']] 
    X = df[['income']] 

    return X, Y, Z

def GLM(y, Z, X):
    n = y.shape[0]
    k = Z.shape[1] + 1
    p = X.shape[1] 

    # reduced model: y = Z * lambda
    W = torch.cat((torch.ones(n).reshape(-1, 1), Z), dim=1)
    lda = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(W.T, W)), W.T), y)
    SSE0 = torch.sum(torch.square(y - torch.matmul(W, lda)))

    # full model: y = Z * lambda + X * beta
    W = torch.cat((torch.ones(n).reshape(-1, 1), Z, X), dim=1)
    # gamma: fitted regression parameters of the full model, where gamma = (lambda, beta)
    gamma = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(W.T, W)), W.T), y)
    SSE1 = torch.sum(torch.square(y - torch.matmul(W, gamma)))

    # F-statistics and its degrees of freedom
    # As F stat is bigger, X is more important to predict y
    F = ((SSE0 - SSE1)/p)/(SSE1/(n-p-k))

    # small pvalue -> favor the full model
    pval = 1 - f.cdf(F, p, n-p-k)

    return F, pval, SSE0, SSE1

def run_GLM(df, race):
    X, Y, Z = get_XYZ(df, race)

    X = torch.tensor(X.values.astype(np.float64)) # n * p
    Z = torch.tensor(Z.values.astype(np.float64)) # n * k
    glm_fstat, glm_pval, glm_Rsse, glm_Fsse = [], [], [], []

    for y_col in Y.columns:
        y = torch.tensor(Y[y_col].values)/1.0
        F, pval, Rsse, Fsse = GLM(y, Z, X)
        glm_fstat.append(F.item())
        glm_pval.append(pval)
        glm_Rsse.append(Rsse.item())
        glm_Fsse.append(Fsse.item())

    return glm_fstat, glm_pval, glm_Rsse, glm_Fsse
    