import sys
import os
import random
import math
import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import copy
from typing import Optional, List


from .utils import toNP
from .geometry import to_basis, from_basis

from .nn import deltaconv
from .diffgeometry.grad_div_mls import build_grad_div, build_tangent_basis, estimate_basis
from .diffgeometry.operators import curl, norm, I_J, hodge_laplacian, laplacian
from .nn.mlp import MLP, VectorMLP

from torch_geometric.nn import knn_graph

class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes 
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)
        

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        #print(self.diffusion_time.data)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec, evecs)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.
    
    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots 
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if(self.with_gradient_rotations):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)
        
        #added more the stablize training
        self.v_bias = torch.nn.Parameter(torch.Tensor(self.C_inout))
        torch.nn.init.uniform_(self.v_bias, -1e-4, 1e-4)
        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
            vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
        else:
            vectorsBreal = self.A(vectors[...,0])
            vectorsBimag = self.A(vectors[...,1])

        dots = vectorsA[...,0] * vectorsBreal + vectorsA[...,1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True, 
                 diffusion_method='spectral',
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        
        self.MLP_C = 2*self.C_width
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class DiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices', mlp_hidden_dims=None, dropout=True, 
                       with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):   
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(C_width = C_width,
                                      mlp_hidden_dims = mlp_hidden_dims,
                                      dropout = dropout,
                                      diffusion_method = diffusion_method,
                                      with_gradient_features = with_gradient_features, 
                                      with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)
        
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


class DeltaDiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True, 
                 diffusion_method='spectral',
                 vector=True,
                 depth = 1,
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(DeltaDiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        
        self.MLP_C = 5*self.C_width
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        if vector:
            self.v_mlp = VectorMLP([C_width * 4 + C_width * 2] + [C_width] * depth)
        else:
            self.v_mlp = None
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY, v, grad, div):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, (div @ v).unsqueeze(0), (curl(v, div)).unsqueeze(0), (norm(v)).unsqueeze(0), x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, (div @ v).unsqueeze(0), (curl(v, div)).unsqueeze(0), (norm(v)).unsqueeze(0)), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in


        if self.v_mlp is not None:
            # Apply operators and concatenate.
            v_cat = torch.cat([v, hodge_laplacian(v, grad, div), grad @ (x0_out.squeeze(0))], dim=1)
            # Combine the operators and their 90-degree rotated variants (I_J) with an MLP.
            v = self.v_mlp(I_J(v_cat))

        return x0_out, v
    
class DeltaDiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices', mlp_hidden_dims=None, dropout=True, 
                       with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):   
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DeltaDiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        #hyper parameter
        self.k = 20
        # Operator construction
        # ---------------------
        self.grad_regularizer=1e-3
        self.grad_kernel_width=1

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DeltaDiffusionNetBlock(C_width = C_width,
                                      mlp_hidden_dims = mlp_hidden_dims,
                                      dropout = dropout,
                                      diffusion_method = diffusion_method,
                                      with_gradient_features = with_gradient_features, 
                                      with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, x_in, mass, pos = None, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            #if pos != None: pos = pos.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        

        batch = None
        # Create a kNN graph, which is used to:
        # 1) Perform maximum aggregation in the scalar stream.
        # 2) Approximate the gradient and divergence oeprators
        #edge_index = knn_graph(pos, self.k, loop=True, flow='target_to_source')

        # Use the normals provided by the data or estimate a normal from the data.
        #   It is advised to estimate normals as a pre-transform.
        # Note: the x_basis and y_basis are referred to in the DeltaConv paper as e_u, and e_v, respectively.
        # Wherever x and y are used to denote tangential coordinates, they can be interchanged with u and v. 
        with torch.no_grad():
            edge_index_normal = knn_graph(pos, 30, loop=True, flow='target_to_source')
            # When normal orientation is unknown, we opt for a locally consistent orientation.
            normal, x_basis, y_basis = estimate_basis(pos, edge_index_normal, orientation=pos)


            # Build the gradient and divergence operators.
            # grad and div are two sparse matrices in the form of SparseTensor.
            grad, div = build_grad_div(pos, normal, x_basis, y_basis, edge_index_normal, batch, kernel_width=self.grad_kernel_width, regularizer= self.grad_regularizer)

        # Apply the first linear layer
        x = self.first_lin(x_in)
        v = grad @ (x.squeeze(0))
        v = v
        # Apply each of the blocks
        for b in self.blocks:
            x, v = b(x, mass, L, evals, evecs, gradX, gradY, v, grad, div)
        
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out




class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=False):
        super().__init__()
        #[N, B, C]
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
       
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)

        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
       

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(
                    cdim, dtype=torch.float32, device=xyz.device
                )
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
    
        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                out = self.get_fourier_embeddings(
                    xyz, num_channels, input_range
                )
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

