# -------------------------------------------------------------------------------------
# Graph Structured Network for Image-Text Matching implementation based on
# https://arxiv.org/abs/2004.00277.
# "Graph Structured Network for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang
#
# Writen by Chunxiao Liu, 2020
# -------------------------------------------------------------------------------------
"""GraphConvolution Layer"""

import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class ImageQueryGraphConvolution(Module):
    '''
    Implementation of: https://arxiv.org/pdf/1611.08402.pdf where we consider
    a fixed sized neighbourhood of nodes for each feature
    '''

    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 n_kernels,
                 coordinate_dim,
                 bias=False):
        super(ImageQueryGraphConvolution, self).__init__()
        '''
        ## Variables:
        - in_feat_dim: dimensionality of input features
        - out_feat_dim: dimensionality of output features
        - n_kernels: number of Gaussian kernels to use
        - coordinate_dim : dimensionality of the pseudo coordinates
        - bias: whether to add a bias to convolutional kernels
        '''

        # Set parameters
        self.n_kernels = n_kernels
        self.coordinate_dim = coordinate_dim
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.bias = bias

        # Convolution filters weights
        self.conv_weights = nn.ModuleList([nn.Linear(
            in_feat_dim, out_feat_dim // n_kernels, bias=bias) for i in range(n_kernels)])

        # Parameters of the Gaussian kernels
        self.mean_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.mean_theta = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_theta = Parameter(torch.Tensor(n_kernels, 1))

        self.init_parameters()

    def init_parameters(self):
        # Initialise Gaussian parameters
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.mean_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0.0, 1.0)
        self.precision_rho.data.uniform_(0.0, 1.0)

    def forward(self, neighbourhood_features, neighbourhood_pseudo_coord):
        '''
        ## Inputs:
        - neighbourhood_features (batch_size, K, neighbourhood_size, in_feat_dim)
        - neighbourhood_pseudo_coord (batch_size, K, neighbourhood_size, coordinate_dim)
        ## Returns:
        - convolved_features (batch_size, K, neighbourhood_size, out_feat_dim)
        '''

        # set parameters
        batch_size = neighbourhood_features.size(0)
        K = neighbourhood_features.size(1)
        neighbourhood_size = neighbourhood_features.size(2)

        # compute pseudo coordinate kernel weights
        weights = self.get_gaussian_weights(neighbourhood_pseudo_coord)

        weights = weights.view(
            batch_size * K, neighbourhood_size, self.n_kernels)

        # compute convolved features
        neighbourhood_features = neighbourhood_features.view(
            batch_size * K, neighbourhood_size, -1)
        convolved_features = self.convolution(neighbourhood_features, weights)
        convolved_features = convolved_features.view(-1, K, self.out_feat_dim)

        return convolved_features

    def get_gaussian_weights(self, pseudo_coord):
        '''
        ## Inputs:       
        - pseudo_coord (batch_size, K, K, pseudo_coord_dim)
        ## Returns:
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        '''

        # compute rho weights
        diff = (pseudo_coord[:, :, :, 0].contiguous(
        ).view(-1, 1) - self.mean_rho.view(1, -1))**2
        weights_rho = torch.exp(-0.5 * diff /
                                (1e-14 + self.precision_rho.view(1, -1)**2))

        # compute theta weights
        first_angle = torch.abs(pseudo_coord[:, :, :, 1].contiguous(
        ).view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle)**2)
                                  / (1e-14 + self.precision_theta.view(1, -1)**2))

        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0

        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        return weights

    def convolution(self, neighbourhood, weights):
        '''
        ## Inputs:
        - neighbourhood (batch_size*K, neighbourhood_size, in_feat_dim)
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        ## Returns:
        - convolved_features (batch_size*K, out_feat_dim)
        '''
        # patch operator
        weighted_neighbourhood = torch.bmm(
            weights.transpose(1, 2), neighbourhood)

        # convolutions
        weighted_neighbourhood = [self.conv_weights[i](
            weighted_neighbourhood[:, i]) for i in range(self.n_kernels)]
        convolved_features = torch.cat(
            [i.unsqueeze(1) for i in weighted_neighbourhood], dim=1)
        convolved_features = convolved_features.view(-1, self.out_feat_dim)

        return convolved_features


class TextQueryGraphConvolution(Module):
    '''
    Implementation of: https://arxiv.org/pdf/1611.08402.pdf where we consider
    a fixed sized neighbourhood of nodes for each feature
    '''

    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 n_kernels,
                 bias=False):
        super(TextQueryGraphConvolution, self).__init__()
        '''
        ## Variables:
        - in_feat_dim: dimensionality of input features
        - out_feat_dim: dimensionality of output features
        - n_kernels: number of Gaussian kernels to use
        - coordinate_dim : dimensionality of the pseudo coordinates
        - bias: whether to add a bias to convolutional kernels
        '''

        # Set parameters
        self.n_kernels = n_kernels
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.bias = bias

        # Convolution filters weights
        self.conv_weights = nn.ModuleList([nn.Linear(
            in_feat_dim, out_feat_dim // n_kernels, bias=bias) for i in range(n_kernels)])

        # Parameters of the Gaussian kernels
        self.params = Parameter(torch.Tensor(n_kernels, 1))

        self.init_parameters()

    def init_parameters(self):
        # Initialise Gaussian parameters
        self.params.data.uniform_(-1.0, 1.0)

    def forward(self, neighbourhood_features, neighbourhood_weights):
        '''
        ## Inputs:
        - neighbourhood_features (batch_size, K, neighbourhood_size, in_feat_dim)
        - neighbourhood_weights (batch_size, n_word, n_word, 1)
        ## Returns:
        - convolved_features (batch_size, K, neighbourhood_size, out_feat_dim)
        '''

        # set parameters
        batch_size = neighbourhood_features.size(0)
        K = neighbourhood_features.size(1)
        neighbourhood_size = neighbourhood_features.size(2)

        # compute neighborhood kernel weights
        weights = self.compute_weights(neighbourhood_weights)
        weights = weights.view(
            batch_size * K, neighbourhood_size, self.n_kernels)

        # compute convolved features
        neighbourhood_features = neighbourhood_features.view(
            batch_size * K, neighbourhood_size, -1)
        convolved_features = self.convolution(neighbourhood_features, weights)
        convolved_features = convolved_features.view(-1, K, self.out_feat_dim)

        return convolved_features

    def compute_weights(self, neighbourhood_weights):
        '''
        ## Inputs:       
        - neighbourhood_weights (batch_size, n_word, n_word, 1)
        ## Returns:
        - weights (batch_size*n_word, n_word, n_kernels)
        '''
        batch_size = neighbourhood_weights.size(0)
        n_word = neighbourhood_weights.size(1)

        weights = neighbourhood_weights.view(-1, 1) * self.params.view(1, -1)
        weights = weights.view(batch_size * n_word, n_word, -1)
        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        return weights

    def convolution(self, neighbourhood, weights):
        '''
        ## Inputs:
        - neighbourhood (batch_size*K, neighbourhood_size, in_feat_dim)
        - weights (batch_size*K, neighbourhood_size, n_kernels)
        ## Returns:
        - convolved_features (batch_size*K, out_feat_dim)
        '''
        # patch operator
        weighted_neighbourhood = torch.bmm(
            weights.transpose(1, 2), neighbourhood)

        # convolutions
        weighted_neighbourhood = [self.conv_weights[i](
            weighted_neighbourhood[:, i]) for i in range(self.n_kernels)]
        convolved_features = torch.cat(
            [i.unsqueeze(1) for i in weighted_neighbourhood], dim=1)
        convolved_features = convolved_features.view(-1, self.out_feat_dim)

        return convolved_features
