import numpy as np
from scipy import special as sci
import math
import torch
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from pygsp.graphs.nngraphs import nngraph

from .laplacian import prepare_laplacian
from .pooling import HealpixPooling, MixedPooling, SpatialPooling, IdentityPooling



class HealpixSampling:
    """Graph Spherical sampling class.
    """
    def __init__(self, n_side, depth, patch_size, sh_degree=None, pooling_mode='average', pooling_name='mixed'):
        """Initialize the sampling class.
        Args:
            n_side (int): Healpix resolution
            depth (int): Depth of the encoder
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            pooling_mode (str, optional): specify the mode for pooling/unpooling.
                                            Can be max or average. Defaults to 'average'.
        """
        assert math.log(n_side, 2).is_integer()
        assert n_side / (2**(depth-1)) >= 1

        G = SphereHealpix(n_side, nest=True, k=8) # Highest resolution sampling
        self.sampling = Sampling(G.coords, sh_degree)
        print(self.sampling.S2SH.shape[1], (sh_degree+1)*(sh_degree//2+1))
        assert self.sampling.S2SH.shape[1] == (sh_degree+1)*(sh_degree//2+1)
        
        self.laps, self.vec = self.get_healpix_laplacians(n_side, depth, laplacian_type="normalized", neighbor=8, pooling_name=pooling_name)
        self.pooling = self.get_healpix_poolings(depth, pooling_mode, patch_size, pooling_name)
    
    def get_healpix_laplacians(self, starting_nside, depth, laplacian_type, neighbor=8, pooling_name='mixed'):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        laps = []
        vec = []
        if not pooling_name in ['spatial', 'spatial_vec', 'spatial_sh']:
            for i in range(depth):
                n_side = starting_nside//(2**i) # Get resolution of the grid at depth i
                if n_side>0:
                    G = SphereHealpix(n_side, nest=True, k=neighbor) # Construct Healpix Graph at resolution n_side
                else:
                    G = SphereHealpix(n_side, nest=True, k=8) # Construct Healpix Graph at resolution n_side
                G.compute_laplacian(laplacian_type) # Compute Healpix laplacian
                laplacian = prepare_laplacian(G.L) # Get Healpix laplacian
                laps.append(laplacian)
                vec.append(G.coords)
        elif pooling_name in ['spatial']:
            n_side = starting_nside
            if n_side>0:
                G = SphereHealpix(n_side, nest=True, k=neighbor) # Construct Healpix Graph at resolution n_side
            else:
                G = SphereHealpix(n_side, nest=True, k=8) # Construct Healpix Graph at resolution n_side
            G.compute_laplacian(laplacian_type) # Compute Healpix laplacian
            laplacian = prepare_laplacian(G.L) # Get Healpix laplacian
            for i in range(depth):
                laps.append(laplacian)
                vec.append(G.coords)
        else:
            for i in range(depth):
                laps.append(torch.ones(1, 1))
                vec.append(np.array([[1, 0, 0]]))
        return laps[::-1], vec[::-1]

    def get_healpix_poolings(self, depth, pooling_mode, patch_size, pooling_name):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            pooling_mode (str): specify the mode for pooling/unpooling. Can be max or average. Defaults to 'average'.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        
        poolings = []
        if pooling_name=='mixed':
            for _ in range(depth-1):
                if patch_size != 1:
                    print(patch_size)
                    stride = (-(patch_size%2) + 2, -(patch_size%2) + 2, -(patch_size%2) + 2)
                    kernel_size_spa = ((patch_size>1) + 1, (patch_size>1) + 1, (patch_size>1) + 1)
                    pool = MixedPooling(mode=pooling_mode, kernel_size_spa=kernel_size_spa, stride=stride)
                    patch_size = int(((patch_size - 2) / 2) * (patch_size%2) + patch_size / 2)
                else:
                    pool = HealpixPooling(mode=pooling_mode)
                poolings.append(pool)
        elif pooling_name in ['spherical']:
            for _ in range(depth-1):
                pool = HealpixPooling(mode=pooling_mode)
                poolings.append(pool)
        elif pooling_name in ['spatial', 'spatial_vec', 'spatial_sh']:
            for _ in range(depth-1):
                pool = SpatialPooling(mode=pooling_mode)
                poolings.append(pool)
        elif pooling_name=='muller':
            for _ in range(depth-1):
                pool = IdentityPooling()
                poolings.append(pool)
        return poolings[::-1]


class BvecSampling:
    """Graph Spherical sampling class.
    """
    def __init__(self, bvec, depth, image_size, sh_degree=None, pooling_mode='average'):
        """Initialize the sampling class.
        Args:
            n_side (int): Healpix resolution
            depth (int): Depth of the encoder
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            pooling_mode (str, optional): specify the mode for pooling/unpooling.
                                            Can be max or average. Defaults to 'average'.
        """

        G = nngraph.NNGraph(bvec, k=10) # Highest resolution sampling
        self.sampling = Sampling(G.coords, sh_degree)
        print(self.sampling.S2SH.shape[1], (sh_degree+1)*(sh_degree//2+1))
        assert self.sampling.S2SH.shape[1] == (sh_degree+1)*(sh_degree//2+1)
        
        self.laps = self.get_bvec_laplacians(G, depth, laplacian_type="normalized")
        self.pooling = self.get_bvec_poolings(depth, pooling_mode, image_size)
    
    def get_bvec_laplacians(self, G, depth, laplacian_type):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        laps = []
        G.compute_laplacian(laplacian_type) # Compute laplacian
        laplacian = prepare_laplacian(G.L) # Get laplacian
        for i in range(depth):
            laps.append(laplacian)
        return laps[::-1]

    def get_bvec_poolings(self, depth, pooling_mode, image_size):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            pooling_mode (str): specify the mode for pooling/unpooling. Can be max or average. Defaults to 'average'.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        
        poolings = []
        for i in range(depth-1):
#            pool = torch.nn.Identity()
            if image_size != 1:
                print(image_size)
                stride = (-(image_size%2) + 2, -(image_size%2) + 2, -(image_size%2) + 2)
                kernel_size_spa = ((image_size>1) + 1, (image_size>1) + 1, (image_size>1) + 1)
                pool = SpatialPooling(mode=pooling_mode, kernel_size_spa=kernel_size_spa, stride=stride)
                image_size = int(((image_size - 2) / 2) * (image_size%2) + image_size / 2)
            else:
                pool = IdentityPooling()
            poolings.append(pool)
        return poolings[::-1]
    
class ShellSampling:
    """Shell Spherical sampling class.
    """
    def __init__(self, vectors_path, shell_path, sh_degree=None, max_sh_degree=None):
        """Initialize the sampling class.
        Args:
            vectors_path (str): Path of the shell sampling vectors (bvecs)
            shell_path (str): Path of the shell sampling shells (bvals)
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            max_sh_degree (int, optional): Max Spherical harmonic degree of the sampling if sh_degree is None
        """
        # Load sampling
        vectors = np.loadtxt(vectors_path)
        shell = np.loadtxt(shell_path)
        if vectors.shape[0] == 3:
            vectors = vectors.T
        
        assert shell.shape[0] == vectors.shape[0]
        assert vectors.shape[1] == 3
        vectors[:, 0] = -vectors[:, 0] # NIFTI FILE HAS STRIDE -1,2,3,4

        self.vectors = vectors # V x 3
        self.shell = shell # V

        # Separate shells
        shell_values, shell_inverse, shell_counts = np.unique(self.shell, return_inverse=True, return_counts=True)
        self.shell_values = shell_values # S
        self.shell_inverse = shell_inverse # V
        self.shell_counts = shell_counts # S

        # Save multi-shell sampling
        self.sampling = []
        for s in self.shell_values:
            vertice = self.vectors[self.shell == s] # V_s x 3
            s_sampling = Sampling(vertice, sh_degree, max_sh_degree, s==0)
            self.sampling.append(s_sampling)


class Sampling:
    """Spherical sampling class.
    """

    def __init__(self, vectors, sh_degree=None, max_sh_degree=None, constant=False, symmetric=True):
        """Initialize symmetric sampling class.
        Args:
            vectors (np.array): [V x 3] Sampling position on the unit sphere (bvecs)
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            max_sh_degree (int, optional): Max Spherical harmonic degree of the sampling if sh_degree is None
            constant (bool, optional): In the case of a shell==0
        """
        # Load sampling
        assert vectors.shape[1] == 3
        self.vectors = vectors # V x 3

        # Compute sh_degree
        if sh_degree is None:
            sh_degree = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4) # We want the number of SHC to be at most the number of vectors
            if not max_sh_degree is None:
                sh_degree = min(sh_degree, max_sh_degree)
        if constant:
            self.S2SH = np.ones((vectors.shape[0], 1)) * math.sqrt(4*math.pi) # V x 1
            self.SH2S = np.zeros(((sh_degree+1)*(sh_degree//2+1), vectors.shape[0])) # (sh_degree+1)(sh_degree//2+1) x V 
            self.SH2S[0] = 1 / math.sqrt(4*math.pi)
        else:
            # Compute SH matrices
            _, self.SH2S = self.sh_matrix(sh_degree, vectors, with_order=1) # (sh_degree+1)(sh_degree//2+1) x V 
            
            # We can't recover more SHC than the number of vertices:
            sh_degree_s2sh = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4)
            sh_degree_s2sh = min(sh_degree_s2sh, sh_degree)
            if not max_sh_degree is None:
                sh_degree_s2sh = min(sh_degree_s2sh, max_sh_degree)
            self.S2SH, _ = self.sh_matrix(sh_degree_s2sh, vectors, with_order=1) # V x (sh_degree_s2sh+1)(sh_degree_s2sh//2+1)

    def sh_matrix(self, sh_degree, vectors, with_order):
        return _sh_matrix(sh_degree, vectors, with_order)

def _sh_matrix(sh_degree, vector, with_order=1, symmetric=True):
    """
    Create the matrices to transform the signal into and from the SH coefficients.

    A spherical signal S can be expressed in the SH basis:
    S(theta, phi) = SUM c_{i,j} Y_{i,j}(theta, phi)
    where theta, phi are the spherical coordinates of a point
    c_{i,j} is the spherical harmonic coefficient of the spherical harmonic Y_{i,j}
    Y_{i,j} is the spherical harmonic of order i and degree j

    We want to find the coefficients c from N known observation on the sphere:
    S = [S(theta_1, phi_1), ... , S(theta_N, phi_N)]

    For this, we use the matrix
    Y = [[Y_{0,0}(theta_1, phi_1)             , ..., Y_{0,0}(theta_N, phi_N)                ],
        ................................................................................... ,
        [Y_{sh_order,sh_order}(theta_1, phi_1), ... , Y_{sh_order,sh_order}(theta_N, phi_N)]]

    And:
    C = [c_{0,0}, ... , c_{sh_order,sh_order}}

    We can express S in the SH basis:
    S = C*Y


    Thus, if we know the signal SH coefficients C, we can find S with:
    S = C*Y --> This code creates the matrix Y

    If we known the signal Y, we can find C with:
    C = S * Y^T * (Y * Y^T)^-1  --> This code creates the matrix Y^T * (Y * Y^T)^-1

    Parameters
    ----------
    sh_degree : int
        Maximum spherical harmonic degree
    vector : np.array (N_grid x 3)
        Vertices of the grid
    with_order : int
        Compute with (1) or without order (0)
    symmetric : bool
        If use symmetric or all SH basis
    Returns
    -------
    spatial2spectral : np.array (N_grid x N_coef)
        Matrix to go from the spatial signal to the spectral signal
    spectral2spatial : np.array (N_coef x N_grid)
        Matrix to go from the spectral signal to the spatial signal
    """
    if with_order not in [0, 1]:
        raise ValueError('with_order must be 0 or 1, got: {0}'.format(with_order))
    if symmetric and (sh_degree%2)!=0:
        raise ValueError('sh_degree must be even or symmetric must be False, got: {0} - {1}'.format(sh_degree, symmetric))

    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    colats = np.arccos(z)
    lons = np.arctan2(y, x) % (2 * np.pi)
    grid = (colats, lons)
    gradients = np.array([grid[0].flatten(), grid[1].flatten()]).T

    num_gradients = gradients.shape[0]
    if symmetric:
        if with_order == 1:
            num_coefficients = int((sh_degree + 1) * (sh_degree/2 + 1))
        else:
            num_coefficients = sh_degree//2 + 1
    else:
        if with_order == 1:
            num_coefficients = int((sh_degree + 1)**2)
        else:
            num_coefficients = sh_degree + 1

    b = np.zeros((num_coefficients, num_gradients))
    for id_gradient in range(num_gradients):
        id_column = 0
        for id_degree in range(0, sh_degree + 1, int(symmetric + 1)):
            for id_order in range(-id_degree * with_order, id_degree * with_order + 1):
                gradients_phi, gradients_theta = gradients[id_gradient]
                y = sci.sph_harm(np.abs(id_order), id_degree, gradients_theta, gradients_phi)
                if id_order < 0:
                    b[id_column, id_gradient] = np.imag(y) * np.sqrt(2)
                elif id_order == 0:
                    b[id_column, id_gradient] = np.real(y)
                elif id_order > 0:
                    b[id_column, id_gradient] = np.real(y) * np.sqrt(2)
                id_column += 1

    b_inv = np.linalg.inv(np.matmul(b, b.transpose()))
    spatial2spectral = np.matmul(b.transpose(), b_inv)
    spectral2spatial = b
    return spatial2spectral, spectral2spatial
