import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import uniform_ as reset
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, **kwargs):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(258, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 128, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])
        # print(self.conv0.weights1.device)
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def build_mlp(
        input_size: int,
        hidden_layer_sizes: list,
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:

    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    nlayers = len(layer_sizes) - 1
    act = [activation for _ in range(nlayers)]
    act[-1] = output_activation

    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module(f"NN-{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module(f"Act-{i}", act[i]())

    return mlp

class Encoder(nn.Module):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            activation: nn.Module):
        super(Encoder, self).__init__()

        self.mlp = nn.Sequential(
            build_mlp(input_features, [mlp_hidden_dim for _ in range(nmlp_layers)], output_features, activation=activation),
            nn.LayerNorm(output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class adaptDeepONet(nn.Module):
    def __init__(self, branch_size, trunk_size, activation, kernel_initializer, num_outputs):
        super(adaptDeepONet, self).__init__()
        self.model = DeepONet(branch_size, trunk_size, activation, kernel_initializer, num_outputs)
        self.output = num_outputs

    def forward(self, x, boundary):
        grid = self.get_grid(x.shape, x.device)
        # x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        # grid = grid.reshape(grid.shape[0], grid.shape[1]*grid.shape[2], grid.shape[3])
        x = [x, grid]
        output = self.model(x).squeeze(-1)
        output = output.T.unsqueeze(-1)
        # print(output)
        output = output.reshape(x[0].shape[0], x[0].shape[1], x[0].shape[2], 1)
        
        return output
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # assume square grid, compute overall grid size
        gridx = torch.linspace(0, size_x, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, size_y, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim):
        super(DeepONet, self).__init__()
        
        # Branch network
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # The output dimension is typically the same as trunk_output_dim
        )
        
        # Trunk network
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # The output dimension is typically the same as branch_output_dim
        )

    def get_grid(self, x):
        # Assume x is of shape (batch_size, num_dim, num_dim, num_features)
        batch_size, num_dim, _, _ = x.shape
        x_range = torch.linspace(0, 1, num_dim)
        y_range = torch.linspace(0, 1, num_dim)
        grid = torch.meshgrid(x_range, y_range) 
        grid = torch.stack(grid, dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return grid
    
    def forward(self, branch_input, boundary_info):
        # Forward pass through branch and trunk networks
        trunk_input = self.get_grid(branch_input).to(branch_input.device)
        branch_output = self.branch(branch_input)
        trunk_output = self.trunk(trunk_input)
        # print(branch_output.shape, trunk_output.shape)
        
        # Compute dot product between branch and trunk outputs
        # Output is the sum of element-wise product between branch and trunk outputs
        output = branch_output * trunk_output
        
        return output
    

class TEECNet(torch.nn.Module):
    r"""The Taylor-series Expansion Error Correction Network which consists of several layers of a Taylor-series Error Correction kernel.

    Args:
        in_channels (int): Size of each input sample.
        width (int): Width of the hidden layers.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers.
        **kwargs: Additional arguments of :class:'torch_geometric.nn.conv.MessagePassing'
    """
    def __init__(self, in_channels, width, out_channels, num_layers=4, **kwargs):
        super(TEECNet, self).__init__()
        self.num_layers = num_layers

        self.fc1 = nn.Linear(in_channels, width)
        self.kernel = KernelConv(width, width, kernel=PowerSeriesKernel, in_edge=1, num_layers=3, **kwargs)
        # self.kernel_out = KernelConv(width, out_channels, kernel=PowerSeriesKernel, in_edge=5, num_layers=2, **kwargs)
        self.fc_out = nn.Linear(width, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.fc1(x)
        for i in range(self.num_layers):
            # x = F.relu(self.kernel(x, edge_index, edge_attr))
            x = self.kernel(x, edge_index, edge_attr)
        # x = self.kernel_out(x, edge_index, edge_attr)
        x = self.fc_out(x)
        # x = F.tanh(x)
        return x


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
    

class PowerSeriesConv(nn.Module):
    def __init__(self, in_channel, out_channel, num_powers, **kwargs):
        super(PowerSeriesConv, self).__init__()
        self.num_powers = num_powers
        self.conv = nn.Linear(in_channel, out_channel)
        self.root_param = nn.Parameter(torch.Tensor(num_powers))  # Scale per power term
        self.activation = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        # for conv in self.convs:
            # nn.init.xavier_uniform_(conv.weight)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.uniform_(self.root_param, -1, 1)

    def forward(self, x):
        # Initialize output tensor
        x_conv = self.conv(x)
        for i in range(self.num_powers):
            if i == 0:
                x_full = self.root_param[i] * x_conv  # Linear term
            else:
                x_full += self.root_param[i] * self.activation(torch.pow(x_conv, i + 1))

        return x_full
    

class PowerSeriesKernel(nn.Module):
    def __init__(self, num_layers, num_powers, activation=nn.Tanh, **kwargs):
        super(PowerSeriesKernel, self).__init__()
        self.num_layers = num_layers
        self.num_powers = num_powers
        self.activation = activation()
        self.conv0 = PowerSeriesConv(kwargs['in_channel'], 16, num_powers)
        self.convs = nn.ModuleList([PowerSeriesConv(16, 16, num_powers) for _ in range(num_layers)])
        self.conv_out = PowerSeriesConv(16, kwargs['out_channel'], num_powers)
        self.norm = nn.BatchNorm1d(16)

    def forward(self, edge_attr):
        x = self.conv0(edge_attr)
        for i in range(self.num_layers):
            x = self.convs[i](x)
            x = self.norm(x)  # Batch normalization
        x = self.conv_out(x)
        return x


class KernelConv(pyg_nn.MessagePassing):
    r"""
    The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP. In our implementation the kernel is combined via a Taylor expansion of 
    graph edge attributes :math:`\mathbf{e}_{i,j}` and a typical neural operator implementation
    of a DenseNet kernel.

    Args:
        in_channel (int): Size of each input sample (nodal values).
        out_channel (int): Size of each output sample (nodal values).
        kernel (torch.nn.Module): A kernel function that maps edge attributes to
            edge weights.
        in_edge (int): Size of each input edge attribute.
        num_layers (int): Number of layers in the Taylor-series expansion kernel.
    """
    def __init__(self, in_channel, out_channel, kernel, in_edge=5, num_layers=3, **kwargs):
        super(KernelConv, self).__init__(aggr='mean')
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.in_edge = in_edge
        self.root_param = nn.Parameter(torch.Tensor(in_channel, out_channel))
        self.bias = nn.Parameter(torch.Tensor(out_channel))

        self.linear = nn.Linear(in_channel, out_channel)
        # self.kernel = kernel(in_channel=in_edge, out_channel=out_channel**2, num_layers=num_layers, **kwargs)
        self.operator_kernel = DenseNet([in_edge, 32, 64, 128, out_channel**2], nn.LeakyReLU)
        if kwargs['retrieve_weight']:
            self.retrieve_weights = True
            self.weight_k = None
            self.weight_op = None
        else:
            self.retrieve_weights = False

        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.kernel)
        reset(self.linear)
        reset(self.operator_kernel)
        size = self.in_channels
        uniform(size, self.root_param)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)
    
    def message(self, x_i, x_j, pseudo):
        # weight_k = self.kernel(pseudo).view(-1, self.out_channels, self.out_channels)
        weight_op = self.operator_kernel(pseudo).view(-1, self.out_channels, self.out_channels)
        
        x_i = self.linear(x_i)
        x_j = self.linear(x_j)
       
        # x_j_k = torch.matmul((x_j - x_i).unsqueeze(1), weight_k).squeeze(1)
        # x_j_k = weight_k
        x_j_op = torch.matmul(x_j.unsqueeze(1), weight_op).squeeze(1)

        # if self.retrieve_weights:
            # self.weight_k = weight_k
            # self.weight_op = weight_op
        # return x_j_k + x_j_op
        return x_j_op
        # return x_j_k
    
    def update(self, aggr_out, x):
        return aggr_out + torch.mm(x, self.root_param) + self.bias
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    

class NNConv_old(pyg_nn.MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv_old, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    

class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in=1, in_width=3, out_width=3):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, out_width)

    def forward(self, x, edge_index, edge_attr):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x