import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class InfoVAE(nn.Module):
    def __init__(self,nfeat=1000, ncode=5, alpha=0, lambd=10000, nhidden=[128, 35], dropout=0.2):
        super(InfoVAE, self).__init__()
        
        self.ncode = int(ncode)
        self.alpha = float(alpha)
        self.lambd = float(lambd)
        self.dropout = dropout

        self.nhidden = nhidden

        self.enc_dense = []
        self.enc_drop = []
        self.dec_dense = []
        self.dec_drop = []

        # number of neuron each encoding layer
        n = [nfeat,] + nhidden
        for i in range(len(n) - 1):
            self.enc_dense.append(nn.Linear(n[i], n[i+1]))
            if dropout > 0:
                self.enc_drop.append(nn.Dropout(p=dropout))

        # latent space layer: mean and log of variance
        self.mu = nn.Linear(nhidden[-1], ncode)
        self.lv = nn.Linear(nhidden[-1], ncode)

        # number of neuron each decoding layer
        n = [ncode,] + nhidden[::-1]
        for i in range(len(n) - 1):
            self.dec_dense.append(nn.Linear(n[i], n[i+1]))
            if dropout > 0:
                self.dec_drop.append(nn.Dropout(p=dropout))
        self.outp = nn.Linear(nhidden[0], nfeat)
        
    def encode(self, x):
        for i in range(len(self.nhidden)):
            if self.dropout > 0:
                x = self.enc_drop[i](F.leaky_relu(self.enc_dense[i](x)))
            else:
                x = F.leaky_relu(self.enc_dense[i](x))
        mu = self.mu(x)
        logvar = self.lv(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, x):
        for i in range(len(self.nhidden)):
            if self.dropout > 0:
                x = self.dec_drop[i](F.leaky_relu(self.dec_dense[i](x)))
            else:
                x = F.leaky_relu(self.dec_dense[i](x))
        x = self.outp(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        # The example code divides by (dim) here, making <kernel_input> ~ 1/dim
        # excluding (dim) makes <kernel_input> ~ 1
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)#/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)
    
    # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
    def compute_mmd(self, x, y):
        xx_kernel = self.compute_kernel(x,x)
        yy_kernel = self.compute_kernel(y,y)
        xy_kernel = self.compute_kernel(x,y)
        return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2*torch.mean(xy_kernel)
    
    def loss(self, x, weig, epoch):
        recon_x, mu, logvar = self.forward(x)
        MSE = torch.sum(0.5 * weig * (x - recon_x).pow(2))
        
        # KL divergence (Kingma and Welling, https://arxiv.org/abs/1312.6114, Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #return MSE + self.beta*KLD, MSE
                
        # https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        true_samples = Variable(torch.randn(200, self.ncode), requires_grad=False)
        z = self.reparameterize(mu, logvar) #duplicate call
        # compute MMD ~ 1, so upweight to match KLD which is ~ n_batch x n_code
        MMD = self.compute_mmd(true_samples,z) * x.size(0) * self.ncode
        return MSE + (1-self.alpha)*KLD + (self.lambd+self.alpha-1)*MMD, MSE, KLD, MMD