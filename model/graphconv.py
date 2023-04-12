import torch
import math
import numpy as np


class Conv(torch.nn.Module):
    """Building Block with a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_sizeSph=3, kernel_sizeSpa=3, bias=True, conv_name='spherical', isoSpa=True):
        """Initialization.
        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.FloatTensor`): laplacian
            kernel_sizeSph (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1. Defaults to 3.
            kernel_sizeSpa (int): Size of the spatial filter.
            bias (bool): Whether to add a bias term.
            conv_name (str): Name of the convolution, either 'spherical' or 'mixed'
        """
        super(Conv, self).__init__()
        self.register_buffer("laplacian", lap)
        if conv_name == 'spherical':
            self.conv = ChebConv(in_channels, out_channels, kernel_sizeSph, bias)
        elif conv_name == 'mixed':
            self.conv = SO3SE3Conv(in_channels, out_channels, kernel_sizeSph, kernel_sizeSpa, bias, isoSpa=isoSpa)
        elif conv_name in ['spatial', 'spatial_vec', 'spatial_sh']:
            self.conv = SpatialConv(in_channels*lap.shape[0], out_channels*lap.shape[0], kernel_sizeSpa, bias, isoSpa=isoSpa)
        else:
            raise NotImplementedError

    def state_dict(self, *args, **kwargs):
        """! WARNING !
        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        del_keys = []
        for key in state_dict:
            if key.endswith("laplacian"):
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): input [B x Fin x V x X x Y x Z]
        Returns:
            :obj:`torch.tensor`: output [B x Fout x V x X x Y x Z]
        """
        x = self.conv(self.laplacian, x)
        return x


##### CHEB CONV ######
class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
        """
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = cheb_conv

        shape = (kernel_size, in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias[None, :, None, None, None, None]
        return outputs


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    # Get tensor dimensions
    B, Fin, V, X, Y, Z = inputs.shape
    K, Fin, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials + 1

    # Transform to Chebyshev basis
    x0 = inputs.permute(2, 1, 0, 3, 4, 5).contiguous()  # V x Fin x B x X x Y x Z
    x0 = x0.view([V, Fin * B * X * Y * Z])  # V x Fin*B*X*Y*Z
    inputs = project_cheb_basis(laplacian, x0, K) # K x V x Fin*B*X*Y*Z

    # Look at the Chebyshev transforms as feature maps at each vertex
    inputs = inputs.view([K, V, Fin, B, X, Y, Z])  # K x V x Fin x B x X x Y x Z
    inputs = inputs.permute(3, 1, 4, 5, 6, 0, 2).contiguous()  # B x V x X x Y x Z x K x Fin
    inputs = inputs.view([B * V * X * Y * Z, K * Fin])  # B*V*X*Y*Z x K*Fin

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout) # K*Fin x Fout
    inputs = inputs.matmul(weight)  # B*V*X*Y*Z x Fout
    inputs = inputs.view([B, V, X, Y, Z, Fout])  # B x V x X x Y x Z x Fout

    # Get final output tensor
    inputs = inputs.permute(0, 5, 1, 2, 3, 4).contiguous()  # B x Fout x V x X x Y x Z

    return inputs

#### SE3 x SO3 CONV #####
class SO3SE3Conv(torch.nn.Module):
    """Graph convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_sizeSph, kernel_sizeSpa, bias=True, isoSpa=True):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_sizeSph (int): Number of trainable parameters per spherical filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            kernel_sizeSpa (int): Size of the spatial filter.
            bias (bool): Whether to add a bias term.
        """
        super(SO3SE3Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizeSph = kernel_sizeSph
        self.kernel_sizeSpa = kernel_sizeSpa
        self.isoSpa = isoSpa
        self._conv = se3so3_conv

        shape = (out_channels, in_channels, kernel_sizeSph)
        self.weightSph = torch.nn.Parameter(torch.Tensor(*shape))

        if self.isoSpa:
            weight_tmp, ind, distance = self.get_index(kernel_sizeSpa)
            self.register_buffer('weight_tmp', weight_tmp)
            self.ind = ind.reshape(kernel_sizeSpa, kernel_sizeSpa, kernel_sizeSpa)
            shape = (out_channels, in_channels, 1, 1, 1, self.weight_tmp.shape[-1])
        else:
            shape = (out_channels, in_channels, kernel_sizeSpa, kernel_sizeSpa, kernel_sizeSpa)
            
        self.weightSpa = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_sizeSph))
        self.weightSph.data.normal_(0, std)
        std = math.sqrt(2 / (self.in_channels * (self.kernel_sizeSpa)))
        self.weightSpa.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def get_index(self, size):
        x_mid = (size - 1)/2
        x = np.arange(size) - x_mid
        distance = np.sqrt(x[None, None, :]**2 + x[None, :, None]**2 + x[:, None, None]**2)
        unique, ind = np.unique(distance, return_inverse=True)
        weight_tmp = torch.zeros((self.out_channels, self.in_channels, size, size, size, len(unique)))
        for i in range(len(unique)):
            weight_tmp[:, :, :, :, :, i][:, :, torch.Tensor(ind.reshape((size, size, size))==i).type(torch.bool)] = 1
        return weight_tmp, ind, distance

    def forward(self, laplacian, inputs):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        if self.isoSpa:
            weight = torch.sum((self.weight_tmp * self.weightSpa), -1)
            outputs = self._conv(laplacian, inputs, self.weightSph, weight)
        else:
            outputs = self._conv(laplacian, inputs, self.weightSph, self.weightSpa)
        if self.bias is not None:
            outputs += self.bias[None, :, None, None, None, None]
        return outputs


def se3so3_conv(laplacian, inputs, weightSph, weightSpa):
    """SE(3) x SO(3) grid convolution.
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weightSph (:obj:`torch.Tensor`): The spherical weights of the current layer.
        weightSpa (:obj:`torch.Tensor`): The spatial weights of the current layer.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    # Get tensor dimensions
    B, Fin, V, X, Y, Z = inputs.shape
    Fout, Fin, K = weightSph.shape
    Fout, Fin, kX, kY, kZ = weightSpa.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials + 1

    # Transform to Chebyshev basis
    x0 = inputs.permute(2, 1, 0, 3, 4, 5).contiguous()  # V x Fin x B x X x Y x Z
    x0 = x0.view([V, Fin * B * X * Y * Z])  # V x Fin*B*X*Y*Z
    inputs = project_cheb_basis(laplacian, x0, K) # K x V x Fin*B*X*Y*Z

    # Look at the Chebyshev transforms as feature maps at each vertex
    inputs = inputs.view([K, V, Fin, B, X, Y, Z])  # K x V x Fin x B x X x Y x Z
    inputs = inputs.permute(3, 1, 2, 0, 4, 5, 6).contiguous()  # B x V x Fin x K x X x Y x Z
    inputs = inputs.view([B * V, Fin * K, X, Y, Z])  # B*V x Fin*K x X x Y x Z

    # Expand spherical and Spatial filters
    wSph = weightSph.view([Fout, Fin*K, 1, 1, 1]).expand(-1, -1, kX, kY, kZ) # Fout x Fin*K x kX x kY x kZ
    wSpa = weightSpa.repeat_interleave(K, dim=1) # Fout x Fin*K x kX x kY x kZ
    weight = wSph * wSpa # Fout x Fin*K x kX x kY x kZ

    # Convolution
    inputs = torch.nn.functional.conv3d(inputs, weight, padding='same') # B*V x Fout x X x Y x Z

    # Get final output tensor
    inputs = inputs.view([B, V, Fout, X, Y, Z])  # B x V x Fout x X x Y x Z
    inputs = inputs.permute(0, 2, 1, 3, 4, 5).contiguous()  # B x Fout x V x X x Y x Z

    return inputs


#### SpatialConv #####
class SpatialConv(torch.nn.Module):
    """Graph convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_sizeSpa, bias=True, isoSpa=True):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_sizeSph (int): Number of trainable parameters per spherical filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            kernel_sizeSpa (int): Size of the spatial filter.
            bias (bool): Whether to add a bias term.
        """
        super(SpatialConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizeSpa = kernel_sizeSpa
        self.isoSpa = isoSpa

        if self.isoSpa:
            weight_tmp, ind, distance = self.get_index(kernel_sizeSpa)
            self.register_buffer('weight_tmp', weight_tmp)
            self.ind = ind.reshape(kernel_sizeSpa, kernel_sizeSpa, kernel_sizeSpa)
            shape = (out_channels, in_channels, 1, 1, 1, self.weight_tmp.shape[-1])
        else:
            shape = (out_channels, in_channels, kernel_sizeSpa, kernel_sizeSpa, kernel_sizeSpa)
        self.weightSpa = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_sizeSpa))
        self.weightSpa.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def get_index(self, size):
        x_mid = (size - 1)/2
        x = np.arange(size) - x_mid
        distance = np.sqrt(x[None, None, :]**2 + x[None, :, None]**2 + x[:, None, None]**2)
        unique, ind = np.unique(distance, return_inverse=True)
        weight_tmp = torch.zeros((self.out_channels, self.in_channels, size, size, size, len(unique)))
        for i in range(len(unique)):
            weight_tmp[:, :, :, :, :, i][:, :, torch.Tensor(ind.reshape((size, size, size))==i).type(torch.bool)] = 1
        return weight_tmp, ind, distance

    def forward(self, laplacian, inputs):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        redim = False
        if len(inputs.shape)==6:
            redim = True
            B, Fin, V, X, Y, Z = inputs.shape
            inputs = inputs.view(B, Fin * V, X, Y, Z)
        if self.isoSpa:
            weight = torch.sum((self.weight_tmp * self.weightSpa), -1)
            outputs = torch.nn.functional.conv3d(inputs, weight, padding='same') # B x Fout x X x Y x Z
        else:
            outputs = torch.nn.functional.conv3d(inputs, self.weightSpa, padding='same') # B x Fout x X x Y x Z
        
        if self.bias is not None:
            outputs += self.bias[None, :, None, None, None]
        if redim:
            _, _, X, Y, Z = outputs.shape
            outputs = outputs.view(B, -1, V, X, Y, Z)
        return outputs



def project_cheb_basis(laplacian, x0, K):
    """Project vector x on the Chebyshev basis of order K
    \hat{x}_0 = x
    \hat{x}_1 = Lx
    \hat{x}_k = 2*L\hat{x}_{k-1} - \hat{x}_{k-2}
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        x0 (:obj:`torch.Tensor`): The initial data being forwarded. [V x D]
        K (:obj:`torch.Tensor`): The order of Chebyshev polynomials + 1.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev projection.
    """
    inputs = x0.unsqueeze(0)  # 1 x V x D
    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)  # V x D
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)  # 2 x V x D
        for _ in range(2, K):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)  # _ x V x D
            x0, x1 = x1, x2
    return inputs # K x V x D
