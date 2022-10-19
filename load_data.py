import torch
import torch.utils.data as data
import numpy as np
from einops import rearrange

class Dataset(data.Dataset):
    def __init__(self, df, race=False):
        super(Dataset, self).__init__()
        cols = ['ddtidp_' + str(i) for i in range(605, 605+148)]
        Y = df[cols] 
        Z = df[['age_x', 'age_y', 'gender', 'GE', 'Philips', 'SIEMENS', 'White', 'Black', 'Others']] 
        X = df[['income']] 

        X = torch.tensor(X.values.astype(np.float64)).float() # n * p
        Z = torch.tensor(Z.values.astype(np.float64)).float() # n * q   
        Y = torch.tensor(Y.values.astype(np.float64)).float() # n * r

        assert X.shape[0] == Z.shape[0] == Y.shape[0] # num_subjects

        self.X = X
        self.Z = Z
        self.Y = Y
        
    def __getitem__(self, index):
        return self.X[index], self.Z[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def getVarSize(self):
        num_roi, p, q = self.Y.shape[1], self.X.shape[1], self.Z.shape[1]
        return num_roi, p, q

    def getNumData(self):
        return self.X.shape[0]

    def getY(self):
        Y = rearrange(self.Y, 'b r -> r b')
        return Y
        