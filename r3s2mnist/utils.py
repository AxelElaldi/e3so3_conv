import numpy as np
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
import numpy as np
from scipy import special as sci

NORTHPOLE_EPSILON = 1e-3

def get_rot_x(angle):
    rot_x =  np.array([[1, 0            ,              0],
                       [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle),  np.cos(angle)]])
    return rot_x

def get_rot_y(angle):
    rot_y =  np.array([[ np.cos(angle), 0, np.sin(angle)],
                       [ 0            , 1,             0],
                       [-np.sin(angle), 0, np.cos(angle)]])
    return rot_y

def get_rot_z(angle):
    rot_z =  np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle),  np.cos(angle), 0],
                       [0            ,  0            , 1]])
    return rot_z

def get_rot_mat(alpha, beta, gamma, atype='euler'):
    if atype=='euler':
        # Rotation with the ZXZ euler angle
        # alpha: rotation around Z (x-->y)
        # beta: rotation around X' (y'-->z)
        # gamma: rotation around Z' (x'-->y')
        rot_x =  get_rot_z(alpha) 
        rot_y =  get_rot_x(beta) 
        rot_z =  get_rot_z(gamma) 
    elif atype=='cardan':
        # Rotation in the original coodinate system (XYZ)
        # alpha: rotation around Z (x-->y)
        # beta: rotation around Y (z-->x)
        # gamma: rotation around X (y-->z)
        rot_x =  get_rot_x(gamma) 
        rot_y =  get_rot_y(beta) 
        rot_z =  get_rot_z(alpha) 
    else:
        raise ValueError(f'atype must be euler or cardan, got: {atype}')
    rot_mat = rot_z.dot(rot_y.dot(rot_x))
    return rot_mat

def get_rotation_matrix():
    # Draw and apply a random rotation matrix
    alpha = float(np.random.rand() * 2 * np.pi)
    rot = np.random.normal(0, 1, 3)
    rot = rot / np.linalg.norm(rot)
    beta = float(np.arccos(rot[2]))
    gamma = float(np.arctan2(rot[1], rot[0]) % (2 * np.pi))
    R = get_rot_mat(alpha, beta, gamma)
    return R, alpha, beta, gamma


def get_projection_grid(b, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    G = SphereHealpix(b, nest=True, k=20)
    vec = G.coords
    x_, y_, z_ = vec[:, 0], vec[:, 1], vec[:, 2]
    return x_, y_, z_


def project_sphere_on_xy_plane(grid, projection_origin):
    ''' returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere '''

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = - (z - sz) / z
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry


def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        #sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample = np.zeros((signal.shape[0], x.shape[0]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        #sample = np.zeros((x.shape[0], x.shape[1]))
        sample = np.zeros((x.shape[0]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.astype(int)
    iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def project_2d_on_sphere(signal, grid, projection_origin=None):
    ''' '''
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(signal, rx, ry) / 4

    # ensure that only south hemisphere gets projected
    sample *= (grid[2] >= 0).astype(np.float64)

    # rescale signal to [0,1]
    #sample_min = sample.min(axis=1).reshape(-1, 1)
    #sample_max = sample.max(axis=1).reshape(-1, 1)
    #if sample_max - sample_min!=0:
    #sample = (sample - sample_min) / (sample_max - sample_min)
    #sample *= 255
    sample = sample.astype(np.uint8)

    return sample


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