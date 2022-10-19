import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils_data
from einops import rearrange

import GLM
import os
import sys
import time
import math
import pickle
import argparse
import load_data
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import f
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from tensorboardX import SummaryWriter

class CoCoNet(nn.Module):
    def __init__(self, p: int = 1, q: int = 5, num_roi: int = 148):
        super(CoCoNet, self).__init__()
        self.num_roi = num_roi
        self.Reduced_Net = nn.ModuleList([nn.Linear(q, 1, bias=True) for _ in range(self.num_roi)])
        self.Full_Net = nn.ModuleList([nn.Linear(p + q, 1, bias=True) for _ in range(self.num_roi)])
        

    def forward(self, X, Z): 
        # X.shape: b x p
        # Z.shape: b x q
        combined_XZ = torch.cat((X, Z), dim = -1) 
        r_out, f_out = [], []
        for i in range(self.num_roi):
            r_out.append(self.Reduced_Net[i](Z)) 
            f_out.append(self.Full_Net[i](combined_XZ))
        
        r_out, f_out = torch.squeeze(torch.stack(r_out)), torch.squeeze(torch.stack(f_out)) 
    
        return r_out, f_out
        

def train(dataset, args):
    start_time = time.time()
    log_dir = os.path.join(args.save_dir, 'tb', 'CoCoNet_' + args.df + '_' + time.strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(log_dir=log_dir)
    dataloader = utils_data.DataLoader(dataset=dataset, batch_size=args.batch, num_workers=8, shuffle=True)
    num_roi, p, q = dataset.getVarSize()
    n = dataset.getNumData()
    n_iter = int(n * 1.0 // args.batch)
     
    criterion = nn.MSELoss(reduction='none')
    network = CoCoNet(p=p, q=q, num_roi=num_roi).cuda() 
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: args.lr_weight ** epoch)
    pbar = tqdm(range(n_iter), file=sys.stdout, bar_format='{desc}[{elapsed}<{remaining},{rate_fmt}]')
       
    # dfF = args.batch - p - (q+1)  &  dfR = args.batch - (q+1) 
    for epoch in range (args.epoch):
        sseR_all, sseF_all = [0]*num_roi, [0]*num_roi
        for idx, data in zip(pbar, dataloader):
            optimizer.zero_grad()

            X, Z, Y = data
            X, Z, Y = X.cuda(), Z.cuda(), Y.cuda()
            Y = rearrange(Y, 'b r -> r b')
            r_pred, f_pred = network(X, Z) 

            sseR = criterion(r_pred, Y)
            sseF = criterion(f_pred, Y) 

            sseR = torch.sum(sseR, dim=1) 
            sseF = torch.sum(sseF, dim=1) 

            beta = []
            for _, (name, param) in enumerate(network.Full_Net.named_parameters()):
                if 'weight' in name: beta.append(param[0][:p])
            beta = torch.stack(beta) 
            l1_norm = torch.norm(beta, p=1)

            fstat = ((sseR - sseF)/p)/(sseF/(n-p-(q+1)))
            pval = torch.tensor(1 - f.cdf(fstat.cpu().detach().numpy(), p, n-p-(q+1)))        

            loss = args.stat_weight * (1 / torch.sum(torch.div(sseR, sseF))) + args.l1_weight * l1_norm \
            + torch.sum((args.R_weight / (q+1)) * sseR) + torch.sum((args.F_weight / (p+q+1)) * sseF)

            loss.backward()
            optimizer.step()

            print_iter_str = 'Epoch{}/{}'.format(epoch, args.epoch) \
                + ' Iter{}/{}:'.format(idx + 1, n_iter) \
                + ' \tLoss=%.4f' % loss.item() \
                + ' \tlr=%.2e' % optimizer.param_groups[0]['lr'] \
                + ' \tRsse=%.2f' % (torch.sum(sseR)/args.batch).item() \
                + ' \tFsse=%.2f' % (torch.sum(sseF)/args.batch).item() 
            print(print_iter_str)
           
            if epoch == args.epoch -1:
                sseR_all = [sum(x) for x in zip(sseR_all, sseR.tolist())]
                sseF_all = [sum(x) for x in zip(sseF_all, sseF.tolist())]
        scheduler.step()
        
        logger.add_scalar('total loss', loss / len(pbar), epoch)
        logger.add_scalar('L1 norm', l1_norm / len(pbar), epoch)
        sseR = torch.sum(sseR) / num_roi
        sseF = torch.sum(sseF) / num_roi
        logger.add_scalar('R_SSE', sseR / len(pbar), epoch) 
        logger.add_scalar('F_SSE', sseF / len(pbar), epoch) 

    logger.close()

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total Training Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours),int(minutes),seconds))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='custom path xxx')
    parser.add_argument('--save_dir', type=str, default='custom path xxx')
    parser.add_argument('--df', type=str, default='UD')
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_weight', type=float, default=0.95)
    parser.add_argument('--stat_weight', type=float, default=0.1)
    parser.add_argument('--l1_weight', type=float, default=0.1)
    parser.add_argument('--R_weight', type=float, default=1.0)
    parser.add_argument('--F_weight', type=float, default=1.0)
    parser.add_argument('--thres', type=float, default=0.05/148)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    df = pd.read_excel(os.path.join(args.data_path, args.df + '_race.xlsx'))
    dataset = load_data.Dataset(df, args.race)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)     
    train(dataset, args)
