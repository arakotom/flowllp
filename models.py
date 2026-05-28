
#%%
import math
import os
import time

import numpy as np
import ot 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class CNNMnist(nn.Module):
    def __init__(self, num_classes):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNCifarDA(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifarDA, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class FullyConnectedClassifier(nn.Module):
    def __init__(self,n_input, n_output):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, 500)  # First layer: 20 inputs, 64 outputs
        self.fc2 = nn.Linear(500, 200)   # Second layer: 64 inputs, 32 outputs
        self.fc3 = nn.Linear(200, 200)    # Third layer: 32 inputs, 2 outputs (for binary classification)
        self.fc4 = nn.Linear(200,n_output)    # Third layer: 32 inputs, 2 outputs (for binary classification)
        self.relu = nn.LeakyReLU(0.2)          # Activation function

    def forward(self, x, get_features=False):
        x = self.relu(self.fc1(x))     # Apply first layer and ReLU
        x = self.relu(self.fc2(x))     # Apply second layer and ReLU
        x = self.relu(self.fc3(x))     # Apply second layer and ReLU
        xr = x.clone()
        x = self.fc4(x)                 # Apply third layer (output layer)
        if get_features:
            return xr, x
        else:
            return x
        

# llpgan models

class Generator(nn.Module):
    def __init__(self,z_dim = 100,dim=2):
        super(Generator, self).__init__()
        self.fc1 =  nn.Linear(z_dim, 512)
        self.fc2 =  nn.Linear(512, 1024)
        self.fc3 =  nn.Linear(1024, 2048)
        self.act = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(2048, dim)
        self.dim = dim
        
    def forward(self, z):
        out = self.act(self.fc1(z))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = self.fc4(out)
        out = out.view(out.shape[0], self.dim)
        return out

class Discriminator(nn.Module):
    def __init__(self,dim,n_class):
        super(Discriminator, self).__init__()
        self.fc1 =  nn.Linear(dim, 2048)
        self.fc2 =  nn.Linear(2048, 1024)
        self.fc3 =  nn.Linear(1024, 512)
        self.act = nn.LeakyReLU(0.2)
        
        self.fc4 = nn.Linear(512, n_class)

    def forward(self, x, get_features=False):
        out = x.view(x.shape[0], -1)
        out = self.act(self.fc1(out))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        if get_features:
            xr = out.clone()
            out = self.fc4(out)
            return xr, out
        else:        
            out = (self.fc4(out))
            return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(nonlinearity):
    """Helper to return the correct activation layer."""
    if nonlinearity == "lrelu":
        return nn.LeakyReLU(0.2)
    elif nonlinearity == "tanh":
        return nn.Tanh()
    return nn.Identity()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 is_transpose=False, use_bn=False, use_bias=True, activation="lrelu"):
        super().__init__()
        
        # Padding 'same' calculation for PyTorch
        padding = kernel_size // 2
        
        if is_transpose:
            # Note: output_padding is needed for stride 2 to maintain exact dimensions
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                           stride, padding, output_padding=stride-1, bias=use_bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                  stride, padding, bias=use_bias)
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class GeneratorConv(nn.Module):
    def __init__(self, z_dim, channels):
        super().__init__()
        # Equivalent to bn_dense(z, 4*4*512)
        self.fc = nn.Linear(z_dim, 4 * 4 * 512, bias=False)
        self.bn_fc = nn.BatchNorm1d(4 * 4 * 512)
        
        self.conv_t1 = ConvBlock(512, 256, 5, 2, is_transpose=True, use_bn=True, use_bias=False)
        self.conv_t2 = ConvBlock(256, 128, 5, 2, is_transpose=True, use_bn=True, use_bias=False)
        
        # Final layer: no BN, uses Tanh
        self.final_conv = nn.ConvTranspose2d(128, channels, 5, 2, padding=2, output_padding=1)

    def forward(self, z):
        z = z.view(-1, z.size(1))  # Flatten if necessary
        h = F.leaky_relu(self.bn_fc(self.fc(z)), 0.2)
        h = h.view(-1, 512, 4, 4) # Reshape to (Batch, C, H, W)
        h = self.conv_t1(h)
        h = self.conv_t2(h)
        h = torch.tanh(self.final_conv(h))
        return h


class DiscriminatorConv(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.dropout20 = nn.Dropout(0.2)
        self.dropout50 = nn.Dropout(0.5)
        
        # Block 1
        self.c1 = ConvBlock(channels, 64, 3, 1)
        self.c2 = ConvBlock(64, 64, 3, 1)
        self.c3 = ConvBlock(64, 64, 3, 2)
        
        # Block 2
        self.c4 = ConvBlock(64, 128, 3, 1)
        self.c5 = ConvBlock(128, 128, 3, 1)
        self.c6 = ConvBlock(128, 128, 3, 2)
        
        # Block 3
        self.c7 = ConvBlock(128, 256, 3, 1)
        self.c8 = ConvBlock(256, 128, 1, 1) # 1x1 convolution
        self.c9 = ConvBlock(128, 64, 1, 1)
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x,get_features=False):
        h = self.dropout20(x)
        h = self.c3(self.c2(self.c1(h)))
        
        h = self.dropout50(h)
        h = self.c6(self.c5(self.c4(h)))
        
        h = self.dropout50(h)
        h = self.c9(self.c8(self.c7(h)))
        
        # Global Average Pooling (Squeeze equivalent)
        h_X = F.avg_pool2d(h, kernel_size=h.size()[2:]).view(h.size(0), -1)
        if get_features:
            return h_X, self.fc(h_X)
        else:
            logits = self.fc(h_X)
            return logits
    

class DCGAN_Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g=64):
        super(DCGAN_Generator, self).__init__()
        # Input: (batch, z_dim, 1, 1)
        self.gen = nn.Sequential(
            # Layer 1: 1x1 -> 4x4
            nn.ConvTranspose2d(z_dim, features_g * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            # Layer 2: 4x4 -> 8x8
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # Layer 3: 8x8 -> 16x16
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # Layer 4: 16x16 -> 32x32
            nn.ConvTranspose2d(features_g * 2, img_channels, 4, 2, 1),
            nn.Tanh(), # Output scaled between -1 and 1
        )

    def forward(self, x):
        return self.gen(x)

class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels, features_d=64, n_class=10):
        super(DCGAN_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: (batch, img_channels, 32, 32)
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 16x16 -> 8x8
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 8x8 -> 4x4
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 4x4 -> 1x1
            #nn.Conv2d(features_d * 4, 1, 4, 2, 0),
            #nn.Sigmoid(),
            # Layer 4: 4x4 ->
            nn.Conv2d(features_d * 4, 2, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, n_class)

    def forward(self, x, get_features=False):
        features = self.disc(x)
        if get_features:
            return features, self.classifier(features)
        else:
            logits = self.classifier(features)
            return logits

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
