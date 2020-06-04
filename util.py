# -----------------------------------------------------------
# Graph Structured Network for Image-Text Matching implementation based on
# "Stacked Cross Attention for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang
# https://arxiv.org/abs/2004.00277.
#
# Writen by Chunxiao Liu, 2020
# ---------------------------------------------------------------
"""Util"""

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm


class Util:

    @staticmethod
    def l1norm(X, dim, eps=1e-8):
        """L1-normalize columns of X
        """
        norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
        X = torch.div(X, norm)
        return X

    @staticmethod
    def l2norm(X, dim, eps=1e-8):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    @staticmethod
    def inter_relation(K, Q, xlambda):
        """
        Q: (batch, queryL, d)
        K: (batch, sourceL, d)
        return (batch, queryL, sourceL)
        """
        batch_size, queryL = Q.size(0), Q.size(1)
        batch_size, sourceL = K.size(0), K.size(1)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        queryT = torch.transpose(Q, 1, 2)

        attn = torch.bmm(K, queryT)
        attn = nn.LeakyReLU(0.1)(attn)
        attn = Util.l2norm(attn, 2)

        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size * queryL, sourceL)
        attn = nn.Softmax(dim=1)(attn * xlambda)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        return attn

    @staticmethod
    def intra_relation(K, Q, xlambda):
        """
        Q: (n_context, sourceL, d)
        K: (n_context, sourceL, d)
        return (n_context, sourceL, sourceL)
        """
        batch_size, sourceL = K.size(0), K.size(1)
        K = torch.transpose(K, 1, 2).contiguous()
        attn = torch.bmm(Q, K)

        attn = attn.view(batch_size * sourceL, sourceL)
        attn = nn.Softmax(dim=1)(attn * xlambda)
        attn = attn.view(batch_size, sourceL, -1)
        return attn
