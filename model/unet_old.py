from math import sqrt
import torch.nn as nn
import torch
from .graphconv import Conv

class GraphCNNUnet(nn.Module):
    """GCNN Autoencoder.
    """
    def __init__(self, in_channels, out_channels, filter_start, kernel_sizeSph, kernel_sizeSpa, pooling, laps):
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
        self.encoder = Encoder(in_channels, filter_start, kernel_sizeSph, kernel_sizeSpa, pooling, laps)
        self.decoder = Decoder(out_channels, filter_start, kernel_sizeSph, kernel_sizeSpa, pooling, laps)

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
    def __init__(self, in_channels, filter_start, kernel_sizeSph, kernel_sizeSpa, poolings, laps):
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
        self.enc_blocks = [Block(in_channels, filter_start, filter_start, laps[-1], kernel_sizeSph, kernel_sizeSpa, conv_name='mixed')]
        self.enc_blocks += [Block((2**i)*filter_start, (2**(i+1))*filter_start, (2**(i+1))*filter_start, laps[-i-2], kernel_sizeSph, kernel_sizeSpa, conv_name='mixed') for i in range(D-2)]
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.pool = nn.ModuleList([pool.pooling for pool in poolings[::-1]])
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x in_channels x V x X x Y x Z]          
        Returns:
            :obj:`torch.Tensor`: output [B x (2**(D-2))*filter_start x V_encoded x X x Y x Z]
            encoder_features (list): Hierarchical encoding. [B x (2**(i))*filter_start x V_encoded_i x X x Y x Z] for i in [0,D-2]
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
    def __init__(self, out_channels, filter_start, kernel_sizeSph, kernel_sizeSpa, poolings, laps):
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
        self.dec_blocks = [Block((2**(D-2))*filter_start, (2**(D-1))*filter_start, (2**(D-2))*filter_start, laps[0], kernel_sizeSph, kernel_sizeSpa, conv_name='mixed')]
        self.dec_blocks += [Block((2**(D-i))*filter_start, (2**(D-i-1))*filter_start, (2**(D-i-2))*filter_start, laps[i], kernel_sizeSph, kernel_sizeSpa, conv_name='mixed') for i in range(1, D-1)]
        self.dec_blocks += [Block(2*filter_start, filter_start, filter_start, laps[-1], kernel_sizeSph, kernel_sizeSpa, conv_name='mixed')]
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.head = Conv(filter_start, out_channels, laps[-1], kernel_sizeSph, kernel_sizeSpa, conv_name='mixed')
        self.activation = nn.ReLU()
        self.unpool = nn.ModuleList([pool.unpooling for pool in poolings])

    def forward(self, x, encoder_features, indiceSpa, indiceSph):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x (2**(D-2))*filter_start x V_encoded_(D-1) x X x Y x Z]
            encoder_features (list): Hierarchical encoding to be forwarded. [B x (2**(i))*filter_start x V_encoded_i x X x Y x Z] for i in [0,D-2]
        Returns:
            :obj:`torch.Tensor`: output [B x V x out_channels x X x Y x Z]
        """
        x = self.dec_blocks[0](x) # B x (2**(D-2))*filter_start x V_encoded_(D-1) x X x Y x Z
        x = self.unpool[0](x, indiceSpa[-1], indiceSph[-1]) # B x (2**(D-2))*filter_start x V_encoded_(D-2) x X x Y x Z
        x = torch.cat([x, encoder_features[-1]], dim=1) # B x 2*(2**(D-2))*filter_start x V_encoded_(D-2) x X x Y x Z
        for i in range(1, len(self.dec_blocks)-1):
            x = self.dec_blocks[i](x) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-1) x X x Y x Z
            x = self.unpool[i](x, indiceSpa[-1-i], indiceSph[-1-i]) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-2) x X x Y x Z
            x = torch.cat([x, encoder_features[-1-i]], dim=1) # B x 2*(2**(D-i-2))*filter_start x V_encoded_(D-i-2) x X x Y x Z
        x = self.dec_blocks[-1](x) # B x filter_start x V x X x Y x Z
        x = self.activation(self.head(x)) # B x out_channels x V x X x Y x Z
        return x
    

class Block(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, in_ch, int_ch, out_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(Block, self).__init__()
        # Conv 1
        self.conv1 = Conv(in_ch, int_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        self.bn1 = nn.BatchNorm3d(int_ch)
        # Conv 2
        self.conv2 = Conv(int_ch, out_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        self.bn2 = nn.BatchNorm3d(out_ch)
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        B, _, V, X, Y, Z = x.shape
        x = self.activation(self.bn1(self.conv1(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_int_ch x V x X x Y x Z
        x = self.activation(self.bn2(self.conv2(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z
        return x



class Block2(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, in_ch, out_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(Block2, self).__init__()
        # Conv 1
        self.conv1 = Conv(in_ch, 2 * in_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        self.bn1 = nn.BatchNorm3d(2 * in_ch)
        # Conv 2
        self.conv2 = Conv(2 * in_ch, 4 * in_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        self.bn2 = nn.BatchNorm3d(4 * in_ch)
        # Conv 3
        self.conv3 = Conv(4 * in_ch, out_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        self.bn3 = nn.BatchNorm3d(out_ch)
        # Conv 4
        #self.conv2 = Conv(8 * in_ch, out_ch, lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        #self.bn2 = nn.BatchNorm3d(out_ch)
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        B, _, V, X, Y, Z = x.shape
        x = self.activation(self.bn1(self.conv1(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_int_ch x V x X x Y x Z
        x = self.activation(self.bn2(self.conv2(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z
        x = self.activation(self.bn3(self.conv3(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z
        #x = self.activation(self.bn4(self.conv4(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z\
        return x



class Segmentation(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, lap, pool, kernel_sizeSph, kernel_sizeSpa, conv_name, start_filter=8, nvec=1, isoSpa=True, input_features=1, n_class=11):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(Segmentation, self).__init__()
        mul = 1
        if conv_name=='spherical':
            mul = sqrt(5)
        elif isoSpa:
            mul = sqrt(4)
        # Conv 1
        self.conv1 = Conv(nvec*input_features, int(start_filter*mul), lap[0], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa)
        self.bn1 = nn.BatchNorm3d(int(start_filter*mul))
        # Conv 2
        self.conv2 = Conv(int(start_filter*mul), int(2*start_filter*mul), lap[1], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa)
        self.bn2 = nn.BatchNorm3d(int(2*start_filter*mul))
        # Conv 3
        self.conv3 = Conv(int(2*start_filter*mul), int(4*start_filter*mul), lap[2], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa)
        self.bn3 = nn.BatchNorm3d(int(4*start_filter*mul))
        # Conv 4
        #self.conv4 = Conv(80, 160, lap[3], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        #self.bn4 = nn.BatchNorm3d(160)
        # Conv cl
        self.convcl = Conv(int(4*start_filter*mul), n_class, lap[0], kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa)
        # Activation
        self.activation = nn.ReLU()
        self.pooling = pool.pooling
        self.softmax = nn.Softmax(dim=1)
        self.conv_name = conv_name
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        if self.conv_name=='spatial':
            B, F, V, X, Y, Z = x.shape
            x = x.reshape(B, F * V, X, Y, Z)
            x = self.activation(self.bn1(self.conv1(x))) # B x F_int_ch x X x Y x Z
            x = self.activation(self.bn2(self.conv2(x))) # B x F_out_ch x X x Y x Z
            x = self.activation(self.bn3(self.conv3(x))) # B x F_out_ch x X x Y x Z
        else:
            B, _, V, X, Y, Z = x.shape
            x = self.activation(self.bn1(self.conv1(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_int_ch x V x X x Y x Z
            x, _, _ = self.pooling(x)
            B, _, V, X, Y, Z = x.shape
            x = self.activation(self.bn2(self.conv2(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z
            x, _, _ = self.pooling(x)
            B, _, V, X, Y, Z = x.shape
            x = self.activation(self.bn3(self.conv3(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z
            x, _, _ = self.pooling(x)
            #B, _, V, X, Y, Z = x.shape
            #x = self.activation(self.bn4(self.conv4(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z
            #x, _, _ = self.pooling(x)
            x = torch.mean(x, axis=2)
        x = self.convcl(x) # B x F_out_ch x X x Y x Z
        #x = self.softmax(x) # B x F_out_ch x X x Y x Z
        return x


class Segmentation2(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, lap, pool, kernel_sizeSph, kernel_sizeSpa, conv_name, start_filter=8, nvec=1, isoSpa=True, input_features=1, n_class=11):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(Segmentation2, self).__init__()
        mul = 1
        if conv_name=='spherical':
            mul = sqrt(5)
        elif isoSpa:
            mul = sqrt(4)
        # Conv 1
        self.conv1 = Conv(nvec*input_features, int(start_filter*mul), lap[0], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa)
        self.bn1 = nn.BatchNorm3d(int(start_filter*mul))
        # Conv 2
        self.conv2 = Conv(int(start_filter*mul), int(2*start_filter*mul), lap[0], kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa)
        self.bn2 = nn.BatchNorm3d(int(2*start_filter*mul))
        # Conv 3
        self.conv3 = Conv(int(2*start_filter*mul), int(4*start_filter*mul), lap[0], kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa)
        self.bn3 = nn.BatchNorm3d(int(4*start_filter*mul))
        # Conv 4
        #self.conv4 = Conv(80, 160, lap[3], kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name)
        #self.bn4 = nn.BatchNorm3d(160)
        # Conv cl
        self.convcl = Conv(int(4*start_filter*mul), n_class, lap[0], kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa)
        # Activation
        self.activation = nn.ReLU()
        self.pooling = pool.pooling
        self.softmax = nn.Softmax(dim=1)
        self.conv_name = conv_name
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        if self.conv_name=='spatial':
            B, F, V, X, Y, Z = x.shape
            x = x.reshape(B, F * V, X, Y, Z)
            x = self.activation(self.bn1(self.conv1(x))) # B x F_int_ch x X x Y x Z
            x = self.activation(self.bn2(self.conv2(x))) # B x F_out_ch x X x Y x Z
            x = self.activation(self.bn3(self.conv3(x))) # B x F_out_ch x X x Y x Z
        else:
            B, _, V, X, Y, Z = x.shape
            x = self.activation(self.bn1(self.conv1(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_int_ch x V x X x Y x Z
            x, _, _ = self.pooling(x)
            x = torch.mean(x, axis=2)
            x = self.activation(self.bn2(self.conv2(x))) # B x F_out_ch x V x X x Y x Z
            x = self.activation(self.bn3(self.conv3(x))) # B x F_out_ch x V x X x Y x Z
            #B, _, V, X, Y, Z = x.shape
            #x = self.activation(self.bn4(self.conv4(x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_out_ch x V x X x Y x Z
            #x, _, _ = self.pooling(x)
        x = self.convcl(x) # B x F_out_ch x X x Y x Z
        #x = self.softmax(x) # B x F_out_ch x X x Y x Z
        return x