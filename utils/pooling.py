import torch.nn as nn
import torch.nn.functional as F

# HEALPIX
class HealpixPooling:
    """Healpix class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode="average"):
        """Initialize healpix pooling and unpooling objects.
        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == "max":
            self.__pooling = HealpixMaxPool()
            self.__unpooling = HealpixMaxUnpool()
        else:
            self.__pooling = HealpixAvgPool()
            self.__unpooling = HealpixAvgUnpool()

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling

## Max Pooling/Unpooling

class HealpixMaxPool(nn.MaxPool1d):
    """Healpix Maxpooling module
    """

    def __init__(self):
        """Initialization
        """
        super().__init__(kernel_size=4, return_indices=True)

    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch
        Args:
            x (:obj:`torch.tensor`):[B x Fin x V x X x Y x Z]
        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [B x Fin x V_pool x X x Y x Z] and indices of pooled pixels
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X x Y x Z x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X*Y*Z x V
        x, indices_sph = F.max_pool1d(x, self.kernel_size, return_indices=True) # B x Fin*X*Y*Z x V_pool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X x Y x Z x V_pool
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_pool x X x Y x Z
        return x, [0], indices_sph


class HealpixMaxUnpool(nn.MaxUnpool1d):
    """Healpix Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x, indices_spa, indices_sph):
        """calls MaxUnpool1d using the indices returned previously by HealpixMaxPool
        Args:
            tuple(x (:obj:`torch.tensor`) : [B x Fin x V x X x Y x Z]
            indices (int)): indices of pixels equiangular maxpooled previously
        Returns:
            [:obj:`torch.tensor`] -- [B x Fin x V_unpool x X x Y x Z]
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X x Y x Z x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X*Y*Z x V
        x = F.max_unpool1d(x, indices_spa, self.kernel_size) # B x Fin*X*Y*Z x V_unpool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X x Y x Z x V_unpool
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_unpool x X x Y x Z
        return x


## Average Pooling/Unpooling

class HealpixAvgPool(nn.AvgPool1d):
    """Healpix Average pooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V x X x Y x Z]
        Returns:
            tuple((:obj:`torch.tensor`), indices (None)): [B x Fin x V_pool x X x Y x Z] and indices for consistence
            with maxPool
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X x Y x Z x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X*Y*Z x V
        x = F.avg_pool1d(x, self.kernel_size) # B x Fin*X*Y*Z x V_pool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X x Y x Z x V_pool
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_pool x X x Y x Z
        return x, [0], [0]


class HealpixAvgUnpool(nn.Module):
    """Healpix Average Unpooling module
    """

    def __init__(self):
        """initialization
        """
        self.kernel_size = 4
        super().__init__()

    def forward(self, x, indices_spa, indices_sph):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor
        Arguments:
            x (:obj:`torch.tensor`), indices (None): [B x Fin x V x X x Y x Z] and indices for consistence with maxUnPool
        Returns:
            [:obj:`torch.tensor`]: [B x Fin x V_unpool x X x Y x Z]
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X x Y x Z x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X*Y*Z x V
        x = F.interpolate(x, scale_factor=self.kernel_size, mode="nearest") # B x Fin*X*Y*Z x V_unpool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X x Y x Z x V_unpool
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_unpool x X x Y x Z
        return x

# MIXED
class MixedPooling:
    """MixedPooling class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode="average", kernel_size_spa=(2, 2, 2), stride=None):
        """Initialize healpix pooling and unpooling objects.
        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == "max":
            self.__pooling = MixedPoolingMaxPool(kernel_size_spa=kernel_size_spa, stride=stride)
            self.__unpooling = MixedPoolingMaxUnpool(kernel_size_spa=kernel_size_spa, stride=stride)
        else:
            self.__pooling = MixedPoolingAvgPool(kernel_size_spa=kernel_size_spa, stride=stride)
            self.__unpooling = MixedPoolingAvgUnpool(kernel_size_spa=kernel_size_spa, stride=stride)

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling

## Max Pooling/Unpooling
class MixedPoolingMaxPool(nn.Module):
    """MixedPooling Maxpooling module
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x):
        """Forward call the 3d Maxpooling of pytorch
        Args:
            x (:obj:`torch.tensor`):[B x Fin x V x X x Y x Z]
        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [B x Fin x V_pool x X_pool x Y_pool x Z_pool] and indices of pooled pixels
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.view(B, Fin * V, X, Y, Z)
        x, indices_spa = F.max_pool3d(x, self.kernel_size_spa, return_indices=True, stride=self.stride)
        _, _, X, Y, Z = x.shape
        x = x.view(B, Fin, V, X, Y, Z)

        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X_pool x Y_pool x Z_pool x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X*Y*Z x V
        x, indices_sphe = F.max_pool1d(x, self.kernel_size_sph, return_indices=True) # B x Fin*X_pool*Y_pool*Z_pool x V_pool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X_pool x Y_pool x Z_pool x V_pool
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_pool x X_pool x Y_pool x Z_pool

        return x, indices_spa, indices_sphe


class MixedPoolingMaxUnpool(nn.Module):
    """MixedPooling Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x, indices_spa, indices_sph):
        """calls MaxUnpool1d using the indices returned previously by MixedPoolingMaxPool
        Args:
            tuple(x (:obj:`torch.tensor`) : [B x Fin x V x X x Y x Z]
            indices (int)): indices of pixels equiangular maxpooled previously
        Returns:
            [:obj:`torch.tensor`] -- [B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool]
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X x Y x Z x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X*Y*Z x V
        x = F.max_unpool1d(x, indices_sph, self.kernel_size_sph) # B x Fin*X*Y*Z x V_unpool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X x Y x Z x V_unpool

        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_unpool x X x Y x Z
        x = x.view(B, Fin * V, X, Y, Z) # B x Fin*V_unpool x X x Y x Z
        x = F.max_unpool3d(x, indices_spa, self.kernel_size_spa, stride=self.stride) # B x Fin*V_unpool x X_unpool x Y_unpool x Z_unpool
        _, _, X, Y, Z = x.shape
        x = x.view(B, Fin, V, X, Y, Z) # B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool

        return x

## Average Pooling/Unpooling
class MixedPoolingAvgPool(nn.Module):
    """MixedPooling Average pooling module
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V x X x Y x Z]
        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [B x Fin x V_pool x X_pool x Y_pool x Z_pool] and indices for consistence
            with maxPool
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.view(B, Fin * V, X, Y, Z)
        x = F.avg_pool3d(x, self.kernel_size_spa, stride=self.stride)
        _, _, X, Y, Z = x.shape
        x = x.view(B, Fin, V, X, Y, Z)
        indices_spa = [0]

        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X_pool x Y_pool x Z_pool x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X_pool*Y_pool*Z_pool x V
        x = F.avg_pool1d(x, self.kernel_size_sph) # B x Fin*X_pool*Y_pool*Z_pool x V_pool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X_pool x Y_pool x Z_pool x V_pool
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_pool x X_pool x Y_pool x Z_pool
        indices_sph = [0]
        return x, indices_spa, indices_sph


class MixedPoolingAvgUnpool(nn.Module):
    """MixedPooling Average Unpooling module
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x, indices_spa, indices_sph):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V x X x Y x Z]
        Returns:
            [:obj:`torch.tensor`]: [B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool]
        """
        B, Fin, V, X, Y, Z = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # B x Fin x X x Y x Z x V
        x = x.view(B, Fin*X*Y*Z, V) # B x Fin*X*Y*Z x V
        x = F.interpolate(x, scale_factor=self.kernel_size_sph, mode="nearest") # B x Fin*X*Y*Z x V_unpool
        _, _, V = x.shape
        x = x.view(B, Fin, X, Y, Z, V) # B x Fin x X x Y x Z x V_unpool

        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # B x Fin x V_unpool x X x Y x Z
        x = x.view(B, Fin * V, X, Y, Z) # B x Fin*V_unpool x X x Y x Z
        x = F.interpolate(x, size=((X-1)*self.stride[0]+self.kernel_size_spa[0], (Y-1)*self.stride[1]+self.kernel_size_spa[1], (Z-1)*self.stride[2]+self.kernel_size_spa[2]), mode="nearest")
        _, _, X, Y, Z = x.shape
        x = x.view(B, Fin, V, X, Y, Z) # B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool
        return x



# Spatial
class SpatialPooling:
    """SpatialPooling class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode="average", kernel_size_spa=(2, 2, 2), stride=None):
        """Initialize healpix pooling and unpooling objects.
        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == "max":
            self.__pooling = SpatialPoolingMaxPool(kernel_size_spa=kernel_size_spa, stride=stride)
            self.__unpooling = SpatialPoolingMaxUnpool(kernel_size_spa=kernel_size_spa, stride=stride)
        else:
            self.__pooling = SpatialPoolingAvgPool(kernel_size_spa=kernel_size_spa, stride=stride)
            self.__unpooling = SpatialPoolingAvgUnpool(kernel_size_spa=kernel_size_spa, stride=stride)

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling

## Max Pooling/Unpooling
class SpatialPoolingMaxPool(nn.Module):
    """MixedPooling Maxpooling module
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x):
        """Forward call the 3d Maxpooling of pytorch
        Args:
            x (:obj:`torch.tensor`):[B x Fin x V x X x Y x Z]
        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [B x Fin x V_pool x X_pool x Y_pool x Z_pool] and indices of pooled pixels
        """
        redim = False
        if len(x.shape)==6:
            redim = True
            B, Fin, V, X, Y, Z = x.shape
            x = x.view(B, Fin * V, X, Y, Z)
        x, indices_spa = F.max_pool3d(x, self.kernel_size_spa, return_indices=True, stride=self.stride)
        if redim:
            _, _, X, Y, Z = x.shape
            x = x.view(B, Fin, V, X, Y, Z)

        return x, indices_spa, None


class SpatialPoolingMaxUnpool(nn.Module):
    """MixedPooling Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x, indices_spa, indices_sph):
        """calls MaxUnpool1d using the indices returned previously by MixedPoolingMaxPool
        Args:
            tuple(x (:obj:`torch.tensor`) : [B x Fin x V x X x Y x Z]
            indices (int)): indices of pixels equiangular maxpooled previously
        Returns:
            [:obj:`torch.tensor`] -- [B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool]
        """
        redim = False
        if len(x.shape)==6:
            redim = True
            B, Fin, V, X, Y, Z = x.shape
            x = x.view(B, Fin * V, X, Y, Z) # B x Fin*V_unpool x X x Y x Z
        x = F.max_unpool3d(x, indices_spa, self.kernel_size_spa, stride=self.stride) # B x Fin*V_unpool x X_unpool x Y_unpool x Z_unpool
        if redim:
            _, _, X, Y, Z = x.shape
            x = x.view(B, Fin, V, X, Y, Z) # B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool

        return x

## Average Pooling/Unpooling
class SpatialPoolingAvgPool(nn.Module):
    """MixedPooling Average pooling module
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V x X x Y x Z]
        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [B x Fin x V_pool x X_pool x Y_pool x Z_pool] and indices for consistence
            with maxPool
        """
        redim = False
        if len(x.shape)==6:
            redim = True
            B, Fin, V, X, Y, Z = x.shape
            x = x.view(B, Fin * V, X, Y, Z)
        x = F.avg_pool3d(x, self.kernel_size_spa, stride=self.stride)
        if redim:
            _, _, X, Y, Z = x.shape
            x = x.view(B, Fin, V, X, Y, Z)

        return x, None, None


class SpatialPoolingAvgUnpool(nn.Module):
    """MixedPooling Average Unpooling module
    """

    def __init__(self, kernel_size_spa=(2, 2, 2), stride=None):
        """initialization
        """
        self.kernel_size_sph = 4
        self.kernel_size_spa = kernel_size_spa
        if stride is None:
            self.stride = kernel_size_spa
        else:
            self.stride = stride
        super().__init__()

    def forward(self, x, indices_spa, indices_sph):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V x X x Y x Z]
        Returns:
            [:obj:`torch.tensor`]: [B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool]
        """
        redim = False
        if len(x.shape)==6:
            redim = True
            B, Fin, V, X, Y, Z = x.shape
            x = x.view(B, Fin * V, X, Y, Z) # B x Fin*V_unpool x X x Y x Z
        else:
            B, Fin, X, Y, Z = x.shape
        x = F.interpolate(x, size=((X-1)*self.stride[0]+self.kernel_size_spa[0], (Y-1)*self.stride[1]+self.kernel_size_spa[1], (Z-1)*self.stride[2]+self.kernel_size_spa[2]), mode="nearest")
        if redim:
            _, _, X, Y, Z = x.shape
            x = x.view(B, Fin, V, X, Y, Z) # B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool
        return x


# Spatial
class IdentityPooling:
    """SpatialPooling class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode="average", kernel_size_spa=(2, 2, 2), stride=None):
        """Initialize healpix pooling and unpooling objects.
        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        self.__pooling = IdentityPooling_()
        self.__unpooling = IdentityUnPooling_()

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling


class IdentityPooling_(nn.Module):
    """MixedPooling Average Unpooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__()

    def forward(self, x):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V x X x Y x Z]
        Returns:
            [:obj:`torch.tensor`]: [B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool]
        """
        return x, None, None

class IdentityUnPooling_(nn.Module):
    """MixedPooling Average Unpooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__()

    def forward(self, x, indices_spa, indices_sph):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V x X x Y x Z]
        Returns:
            [:obj:`torch.tensor`]: [B x Fin x V_unpool x X_unpool x Y_unpool x Z_unpool]
        """
        return x
