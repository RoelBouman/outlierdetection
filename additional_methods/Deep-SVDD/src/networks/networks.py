import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class network(BaseNet):

    def __init__(self, n_vars, n_layers, shrinkage_factor):
        super().__init__()
        
        layer_sizes = [math.ceil(n_vars * (1-shrinkage_factor)**(i)) for i in range(n_layers+1)]
        self.rep_dim = layer_sizes[-1]
        
        self.layers = []
        
        for i in range(n_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            if i is not n_layers-1:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i+1], eps=1e-04, affine=False))
                self.layers.append(nn.LeakyReLU())
        
        self.encoder = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class auto_encoder(BaseNet):

    def __init__(self, n_vars, n_layers, shrinkage_factor):
        super().__init__()
        
        layer_sizes = [math.ceil(n_vars * (1-shrinkage_factor)**(i)) for i in range(n_layers+1)]
        self.rep_dim = layer_sizes[-1]
        
        #encoder
        self.layers = []
        
        for i in range(n_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            
            if i is not n_layers-1:
                nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('leaky_relu'))
                self.layers.append(nn.BatchNorm1d(layer_sizes[i+1], eps=1e-04, affine=False))
                self.layers.append(nn.LeakyReLU())
        
        self.encoder = nn.Sequential(*self.layers)
        
        #decoder
        
        reverse_layer_sizes = list(reversed(layer_sizes))
        self.layers = []
        
        for i in range(n_layers):
            self.layers.append(nn.Linear(reverse_layer_sizes[i], reverse_layer_sizes[i+1], bias=False))
            if i is not n_layers-1:
                nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('leaky_relu'))
                self.layers.append(nn.BatchNorm1d(reverse_layer_sizes[i+1], eps=1e-04, affine=False))
                self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
