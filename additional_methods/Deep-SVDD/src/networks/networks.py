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

#fix xavier initialisation
class auto_encoder(BaseNet):

    def __init__(self, n_vars, n_layers, shrinkage_factor):
        super().__init__()
        
        layer_sizes = [math.ceil(n_vars * (1-shrinkage_factor)**(i)) for i in range(n_layers+1)]
        self.rep_dim = layer_sizes[-1]
        
        self.layers = []
        
        for i in range(n_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]), bias=False)
            if i is not n_layers-1:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i+1], eps=1e-04, affine=False))
                self.layers.append(F.leaky_relu())

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x
