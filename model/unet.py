from math import sqrt
import torch.nn as nn
import torch
from .blockconstructor import Block, BlockHead

class GraphCNNUnet(nn.Module):
    """GCNN Autoencoder.
    """
    def __init__(self, in_channels, out_channels, filter_start, block_depth, in_depth, kernel_sizeSph, kernel_sizeSpa, poolings, laps, conv_name, isoSpa, keepSphericalDim, vec, n_vec=None):
        """Initialization.
        Args:
            in_channels (int): Number of input channel
            out_channels (int): Number of output channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(GraphCNNUnet, self).__init__()
        self.conv_name = conv_name
        self.encoder = Encoder(in_channels, filter_start, block_depth, kernel_sizeSph, kernel_sizeSpa, poolings, laps, conv_name, isoSpa, vec)
        self.decoder = Decoder(out_channels, filter_start, in_depth, kernel_sizeSph, kernel_sizeSpa, poolings, laps, conv_name, isoSpa, keepSphericalDim, vec, n_vec)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x out_channels x V x X x Y x Z]
        """
        x, enc_ftrs, indiceSpa, indiceSph = self.encoder(x)
        x = self.decoder(x, enc_ftrs, indiceSpa, indiceSph)
        return x


class Encoder(nn.Module):
    """GCNN Encoder.
    """
    def __init__(self, in_channels, filter_start, block_depth, kernel_sizeSph, kernel_sizeSpa, poolings, laps, conv_name, isoSpa, vec):
        """Initialization.
        Args:
            in_channels (int): Number of input channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            poolings (list :obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(Encoder, self).__init__()
        D = len(laps)
        assert D > 1 # Otherwise there is no encoding/decoding to perform
        self.enc_blocks = [Block([in_channels] + [filter_start]*block_depth, laps[-1], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, vec=[vec[-1]] + [vec[-1]]*(block_depth-1) + [vec[-2]])]
        self.enc_blocks += [Block([(2**i)*filter_start] + [(2**(i+1))*filter_start]*block_depth, laps[-i-2], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, vec=[vec[-i-2]] + [vec[-i-2]]*(block_depth-1) + [vec[-i-3]]) for i in range(D-2)]
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.pool = nn.ModuleList([pool.pooling for pool in poolings[::-1]])
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x in_channels x V x X x Y x Z] or [B x in_channels x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x (2**(D-2))*filter_start x V_encoded x X x Y x Z] or [B x (2**(D-2))*filter_start x X x Y x Z]
            encoder_features (list): Hierarchical encoding. [B x (2**(i))*filter_start x V_encoded_i x X x Y x Z] or [B x (2**(i))*filter_start x X x Y x Z] for i in [0,D-2]
        """
        ftrs = []
        indiceSpa = []
        indiceSph = []
        for i, block in enumerate(self.enc_blocks): # len(self.enc_blocks) = D - 1
            x = block(x) # B x (2**(i))*filter_start x V_encoded_i x X_(i) x Y_(i) x Z_(i)
            ftrs.append(x) 
            x, indSpa, indSph = self.pool[i](x) # B x (2**(i))*filter_start x V_encoded_(i+1) x X_(i+1) x Y_(i+1) x Z_(i+1)
            indiceSpa.append(indSpa)
            indiceSph.append(indSph)
        return x, ftrs, indiceSpa, indiceSph


class Decoder(nn.Module):
    """GCNN Decoder.
    """
    def __init__(self, out_channels, filter_start, in_depth, kernel_sizeSph, kernel_sizeSpa, poolings, laps, conv_name, isoSpa, keepSphericalDim, vec, n_vec):
        """Initialization.
        Args:
            out_channels (int): Number of output channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            poolings (list :obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(Decoder, self).__init__()
        D = len(laps)
        assert D > 1 # Otherwise there is no encoding/decoding to perform
        self.dec_blocks = [Block([(2**(D-2))*filter_start] + [(2**(D-1))*filter_start]*in_depth + [(2**(D-2))*filter_start], laps[0], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, vec=[vec[0]] + [vec[0]]*in_depth + [vec[0]])]
        self.dec_blocks += [Block([(2**(D-i))*filter_start] + [(2**(D-i-1))*filter_start]*in_depth + [(2**(D-i-2))*filter_start], laps[i], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, vec=[vec[i-1]] + [vec[i]]*in_depth + [vec[i]]) for i in range(1, D-1)]
        self.dec_blocks += [Block([2*filter_start] + [filter_start]*in_depth + [filter_start], laps[-1], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, vec=[vec[-2]] + [vec[-1]]*in_depth + [vec[-1]])]
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.head = BlockHead([filter_start, out_channels], laps[-1], kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, keepSphericalDim, [vec[-1], vec[-1]], n_vec)
        #self.activation = nn.ReLU()
        self.activation = nn.Softplus()
        self.unpool = nn.ModuleList([pool.unpooling for pool in poolings])

    def forward(self, x, encoder_features, indiceSpa, indiceSph):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x (2**(D-2))*filter_start x V_encoded_(D-1) x X x Y x Z]
            encoder_features (list): Hierarchical encoding to be forwarded. [B x (2**(i))*filter_start x V_encoded_i x X x Y x Z] for i in [0,D-2]
        Returns:
            :obj:`torch.Tensor`: output [B x out_channels x V x X x Y x Z]
        """
        x = self.dec_blocks[0](x) # B x (2**(D-2))*filter_start x V_encoded_(D-1) x X x Y x Z
        x = self.unpool[0](x, indiceSpa[-1], indiceSph[-1]) # B x (2**(D-2))*filter_start x V_encoded_(D-2) x X x Y x Z
        x = torch.cat([x, encoder_features[-1]], dim=1) # B x 2*(2**(D-2))*filter_start x V_encoded_(D-2) x X x Y x Z
        for i in range(1, len(self.dec_blocks)-1):
            x = self.dec_blocks[i](x) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-1) x X x Y x Z
            x = self.unpool[i](x, indiceSpa[-1-i], indiceSph[-1-i]) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-2) x X x Y x Z
            x = torch.cat([x, encoder_features[-1-i]], dim=1) # B x 2*(2**(D-i-2))*filter_start x V_encoded_(D-i-2) x X x Y x Z
        x = self.dec_blocks[-1](x) # B x filter_start x V x X x Y x Z
        x = self.activation(self.head(x)) # B x out_channels (x V) x X x Y x Z
        return x