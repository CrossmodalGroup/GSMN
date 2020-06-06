# -------------------------------------------------------------------------------------
# Graph Structured Network for Image-Text Matching implementation based on
# https://arxiv.org/abs/2004.00277.
# "Graph Structured Network for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang
#
# Writen by Chunxiao Liu, 2020
# -------------------------------------------------------------------------------------
"""Visual Graph and Textual Graph"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import VisualGraphConvolution
from layers import TextualGraphConvolution
from util import Util
import numpy as np


class VisualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 n_kernels=8):
        '''
        ## Variables:
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - n_kernels : number of Gaussian kernels for convolutions
        '''

        super(VisualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # graph convolution layers
        self.vGNN = VisualGraphConvolution(feat_dim, hid_dim, n_kernels, 2)

        # # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))

    def _compute_pseudo(self, bbox):
        '''

        Computes pseudo-coordinates from bounding box centre coordinates

        ## Inputs:
        - bb_centre (batch_size, K, coord_dim)
        - polar (bool: polar or euclidean coordinates)
        ## Returns:
        - pseudo_coord (batch_size, K, K, coord_dim)
        '''
        bb_size = (bbox[:, :, 2:] - bbox[:, :, :2])
        bb_centre = bbox[:, :, :2] + 0.5 * bb_size
        K = bb_centre.size(1)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
        theta = torch.atan2(
            pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
        pseudo_coord = torch.cat(
            (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)

        return pseudo_coord

    def node_level_matching(self, tnodes, vnodes, n_block, xlambda):
        # Node-level matching: find relevant nodes from another modality
        inter_relation = Util.inter_relation(tnodes, vnodes, xlambda)

        # Compute sim with weighted context
        # (batch, n_word, n_region)
        attnT = torch.transpose(inter_relation, 1, 2)
        contextT = torch.transpose(tnodes, 1, 2)  # (batch, dim, n_word)
        weightedContext = torch.bmm(contextT, attnT)  # (batch, dim, n_region)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)  # (batch, n_region, dims)

        # Multi-block similarity
        # (batch, n_region, num_block, dims/num_block)
        qry_set = torch.split(vnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)

        # (batch, n_region, num_block)
        vnode_mvector = Util.cosine_similarity(
            qry_set, ctx_set, dim=-1)

        return vnode_mvector

    def structure_level_matching(self, vnode_mvector, pseudo_coord):
        # (batch, n_region, n_region, num_block)
        batch, n_region = vnode_mvector.size(0), vnode_mvector.size(1)
        neighbor_image = vnode_mvector.unsqueeze(
            2).repeat(1, 1, n_region, 1)

        # Propagate matching vector to neighbors to infer phrase correspondence
        hidden_graph = self.vGNN(neighbor_image, pseudo_coord)
        hidden_graph = hidden_graph.view(batch * n_region, -1)

        # Jointly infer matching score
        sim = self.out_2(self.out_1(hidden_graph).tanh())
        sim = sim.view(batch, -1).mean(dim=1, keepdim=True)

        return sim

    def forward(self, images, captions, bbox, cap_lens, opt):
        similarities = []
        n_block = opt.embed_size // opt.num_block
        n_image, n_caption = images.size(0), captions.size(0)

        pseudo_coord = self._compute_pseudo(bbox).cuda()
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            # --> compute similarity between query region and context word
            # --> (batch, n_region, n_word)
            vnode_mvector = self.node_level_matching(
                cap_i_expand, images, n_block, opt.lambda_softmax)
            v2t_similarity = self.structure_level_matching(
                vnode_mvector, pseudo_coord)

            similarities.append(v2t_similarity)

        similarities = torch.cat(similarities, 1)
        return similarities


class TextualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 n_kernels=8):
        '''
        ## Variables:
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - n_kernels : number of Gaussian kernels for convolutions
        '''

        super(TextualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        # graph convolution layers
        self.tGNN = TextualGraphConvolution(feat_dim, hid_dim, n_kernels, 2)

        # # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))

    def build_sparse_graph(self, dep, lens):
        adj = np.zeros((lens, lens), dtype=np.int)
        for i, pair in enumerate(dep):
            if i == 0 or pair[0] >= lens or pair[1] >= lens:
                continue
            adj[pair[0], pair[1]] = 1
            adj[pair[1], pair[0]] = 1
        adj = adj + np.eye(lens)
        return torch.from_numpy(adj).cuda().float()

    def node_level_matching(self, vnodes, tnodes, n_block, xlambda):

        inter_relation = Util.inter_relation(vnodes, tnodes, xlambda)

        # Compute sim with weighted context
        # (batch, n_region, n_word)
        attnT = torch.transpose(inter_relation, 1, 2)
        contextT = torch.transpose(vnodes, 1, 2)  # (batch, dim, n_region)
        weightedContext = torch.bmm(contextT, attnT)  # (batch, dim, n_word)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)  # (batch, n_word, dims)

        # Multo-block similarity
        # (batch, n_word, num_block, dims/num_block)
        qry_set = torch.split(tnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)

        tnode_mvector = Util.cosine_similarity(
            qry_set, ctx_set, dim=-1)  # (batch, n_word, num_block)
        return tnode_mvector

    def structure_level_matching(self, tnode_mvector, intra_relation, depends, opt):
        # (batch, n_word, 1, num_block)
        tnode_mvector = tnode_mvector.unsqueeze(2)
        batch, n_word = tnode_mvector.size(0), tnode_mvector.size(1)

        if not opt.is_sparse:
            neighbor_nodes = tnode_mvector.repeat(1, 1, n_word, 1)
            neighbor_weights = intra_relation.repeat(batch, 1, 1, 1)
        else:
            # Build adjacency matrix for each text query
            # (1, n_word, n_word, 1)
            adj_mtx = self.build_sparse_graph(depends, n_word)
            adj_mtx = adj_mtx.view(n_word, n_word).unsqueeze(0).unsqueeze(-1)
            # (batch, n_word, n_word, num_block)
            neighbor_nodes = adj_mtx * tnode_mvector
            # (batch, n_word, n_word, 1)
            neighbor_weights = Util.l2norm(adj_mtx * intra_relation, dim=2)
            neighbor_weights = neighbor_weights.repeat(batch, 1, 1, 1)

        hidden_graph = self.tGNN(neighbor_nodes, neighbor_weights)
        hidden_graph = hidden_graph.view(batch * n_word, -1)

        sim = self.out_2(self.out_1(hidden_graph).tanh())
        sim = sim.view(batch, -1).mean(dim=1, keepdim=True)
        return sim

    def forward(self, images, captions, depends, cap_lens, opt):
        n_image = images.size(0)
        n_caption = captions.size(0)
        similarities = []
        n_block = opt.embed_size // opt.num_block
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()

            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            # --> compute similarity between query region and context word
            # --> (batch, n_word, n_region)
            words_sim = Util.intra_relation(
                cap_i, cap_i, opt.lambda_softmax).unsqueeze(-1)
            nodes_sim = self.node_level_matching(
                images, cap_i_expand, n_block, opt.lambda_softmax)
            phrase_sim = self.structure_level_matching(
                nodes_sim, words_sim, depends, opt)

            similarities.append(phrase_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        return similarities
