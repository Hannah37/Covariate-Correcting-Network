from turtle import ycor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils_data
from einops import rearrange


import os
import sys
import time
import argparse
import load_data
import pandas as pd
from tqdm import tqdm
from scipy.stats import f
from tensorboardX import SummaryWriter

''' GLM to Neural Network + Add Lasso (L1 regularizer) to Full Models'''

class Reduced_Net(nn.Module):
  def __init__(self, q: int = 5):
    super(Reduced_Net, self).__init__()
    self.fc = nn.Linear(q, 1, bias=True) 

  def forward(self, Z):
    return self.fc(Z)

class Full_Net(nn.Module):
  def __init__(self, p: int = 2, q: int = 5):
    super(Full_Net, self).__init__()
    self.fc = nn.Linear(p + q, 1, bias=True)

  def forward(self, X, Z):
    combined_XZ = torch.cat((X, Z), dim = -1)
    return self.fc(combined_XZ)
        

def train(dataset, args):
    start_time = time.time()
    log_dir = os.path.join(args.save_dir, 'tb', 'GLMLasso_' + args.df + '_'  + time.strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(log_dir=log_dir)
    dataloader = utils_data.DataLoader(dataset=dataset, batch_size=args.batch, num_workers=8, shuffle=True)
    num_roi, p, q = dataset.getVarSize()
    n = dataset.getNumData()
    n_iter = int(n * 1.0 // args.batch)
     
    criterion = nn.MSELoss(reduction='none')
    Reduced_Nets, Full_Nets = [], []
    for i in range(num_roi):
        Reduced_Nets.append(Reduced_Net(q=q).cuda())
        Full_Nets.append(Full_Net(p=p, q=q).cuda())
    R_opts = [optim.Adam(net.parameters(), lr=args.lr) for net in Reduced_Nets] 
    F_opts = [optim.Adam(net.parameters(), lr=args.lr) for net in Full_Nets]

    R_scheduler, F_scheduler = [], []
    for r_opt, f_opt in zip(R_opts, F_opts):
        r_scheduler = optim.lr_scheduler.LambdaLR(optimizer=r_opt,
                                    lr_lambda=lambda epoch: args.lr_weight ** epoch)
        f_scheduler = optim.lr_scheduler.LambdaLR(optimizer=f_opt,
                            lr_lambda=lambda epoch: args.lr_weight ** epoch)
        R_scheduler.append(r_scheduler)
        F_scheduler.append(f_scheduler)
    pbar = tqdm(range(n_iter), file=sys.stdout, bar_format='{desc}[{elapsed}<{remaining},{rate_fmt}]')

    for epoch in range (args.epoch):
        sseR_all, sseF_all = [0]*num_roi, [0]*num_roi
        for iter_idx, data in zip(pbar, dataloader):
            X, Z, Y = data
            X, Z, Y = X.cuda(), Z.cuda(), Y.cuda()
            Y = rearrange(Y, 'b r -> r b')

            for net_idx, (r_net, f_net, y) in enumerate(zip(Reduced_Nets, Full_Nets, Y)):
                R_opts[net_idx].zero_grad()
                F_opts[net_idx].zero_grad()

                r_pred = r_net(Z)
                f_pred = f_net(X, Z)
                r_pred = r_pred.squeeze()
                f_pred = f_pred.squeeze()

                sseR = criterion(r_pred, y) 
                sseF = criterion(f_pred, y) 
                sseR = torch.sum(sseR) 
                sseF = torch.sum(sseF)     

                sseR.backward()
                sseF.backward()
                R_opts[net_idx].step()
                F_opts[net_idx].step()

                sseR_all[net_idx] += sseR.item()
                sseF_all[net_idx] += sseF.item() 

            print_iter_str = 'Epoch{}/{}'.format(epoch, args.epoch) \
                + ' Iter{}/{}:'.format(iter_idx + 1, n_iter) \
                + ' \tRsse=%.2f' % sseR.item() \
                + ' \tFsse=%.2f' % sseF.item() \
                + ' \tlr=%.2e' % R_opts[0].param_groups[0]['lr'] 
            print(print_iter_str)

            beta = []
            for f_idx, f_net in enumerate(Full_Nets):
                F_opts[f_idx].zero_grad()
                for _, (name, param) in enumerate(f_net.named_parameters()):
                    if 'weight' in name:
                        beta.append(param[0][:p])
            beta = torch.stack(beta) 
            l1_norm = torch.norm(beta, p=1) 
                
            lasso = args.l1_weight * l1_norm
            lasso.backward()
            for f_idx, _ in enumerate(Full_Nets):
                F_opts[f_idx].step()
        
        for r_scheduler, f_scheduler in zip(R_scheduler, F_scheduler):
            r_scheduler.step()
            f_scheduler.step()

        sseR, sseF = torch.tensor(sseR_all), torch.tensor(sseF_all)
        fstat = ((sseR - sseF)/p)/(sseF/(n-p-(q+1)))
        pval = torch.tensor(1 - f.cdf(fstat.cpu().detach().numpy(), p, n-p-(q+1))) 

        sseR = torch.sum(sseR) / num_roi
        sseF = torch.sum(sseF) / num_roi
            
        print_epoch_str = '\nEpoch{}/{}'.format(epoch, args.epoch) \
                + ' \tmax f-stat=%.2f' % torch.max(fstat).item() \
                + ' \tmin f-stat=%.2f' % torch.min(fstat).item()   
        print(print_epoch_str)
        
        logger.add_scalar('R_SSE', sseR / len(pbar), epoch) 
        logger.add_scalar('F_SSE', sseF / len(pbar), epoch) 
        logger.add_scalar('Lasso', lasso / len(pbar), epoch) 
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
    parser.add_argument('--lr_weight', type=float, default=0.95)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--thres', type=float, default=0.05/148)
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    df = pd.read_excel(os.path.join(args.data_path, args.df))
    dataset = load_data.Dataset(df, args.race)

    train(dataset, args)



