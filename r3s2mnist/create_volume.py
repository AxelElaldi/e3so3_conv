import os
import pickle
import gzip
import numpy as np
from utils import get_rotation_matrix
import argparse
from scipy.interpolate import RegularGridInterpolator


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_path',
    required=True,
    help='Root path of the dataset (default: None)',
    type=str
)
parser.add_argument(
    '--prefix',
    required=True,
    help='Prefix of the dataset (default: None)',
    type=str
)
parser.add_argument(
    '--dataset_size',
    default=1000,
    help='Dataset size (default: 1000)',
    type=int
)
parser.add_argument(
    '--grid_size',
    default=16,
    help='Grid size (default: 16)',
    type=int
)
parser.add_argument(
    '--tube_size',
    default=4,
    help='Tube size (default: 4)',
    type=int
)
parser.add_argument(
    '--rotation',
    action='store_true',
    help='Random SO3 rotation (default: False)',
)
parser.add_argument(
    '--fixed_background',
    action='store_true',
    help='Set background to the class 0, otherwise class 10 with random digits (default: False)',
)

args = parser.parse_args()
dataset_path = args.dataset_path
prefix = args.prefix
dataset_size = args.dataset_size
grid_size = args.grid_size
tube_size = args.tube_size
rotation = args.rotation
fixed_background = args.fixed_background

# Dataset volume initialization
ground_truth_label = np.zeros((dataset_size, grid_size, grid_size, grid_size), dtype=int)
ground_truth_relative_position = np.zeros((dataset_size, grid_size, grid_size, grid_size, 3), dtype=int)
ground_truth_grid_rotation = np.zeros((dataset_size, 3, 3))
ground_truth_grid_rotation_angles = np.zeros((dataset_size, 3))
ground_truth_top_left_corner = np.zeros((dataset_size, 9, 2), dtype=int)

# Usefull lists to randomly draw squares
grid_x = np.arange(grid_size)
grid_coord = np.stack((np.meshgrid(grid_x, grid_x, indexing='ij')), axis=-1).reshape((-1, 2)) # List of possible tube coordinates
grid_inds = np.arange(grid_size*grid_size) # Index of possible tube cooridnates

# Usefull lists to keep in memory relative position of a voxel in a tube
tube_x = np.arange(tube_size)
tube_coord = np.stack((np.meshgrid(tube_x, tube_x, grid_x, indexing='ij')), axis=-1) + 1

# Grid coordinate
xx = np.arange(grid_size) - grid_size/2 + ((grid_size+1)%2)*0.5
yy = np.arange(grid_size) - grid_size/2 + ((grid_size+1)%2)*0.5
zz = np.arange(grid_size) - grid_size/2 + ((grid_size+1)%2)*0.5
vol_coord = np.stack(np.meshgrid(xx, yy, zz, indexing='ij'), axis=0)

for created_volume in range(dataset_size):
    ind_class = 0 # While the algorithm could not fit 9 tubes into the volume
    while ind_class!=8:
        # Generated dataset
        vol = np.zeros((grid_size, grid_size, grid_size)) # Generated volume
        rel_pos = np.zeros((grid_size, grid_size, grid_size, 3)) # Relative position of the voxel in a tube
        corner = np.zeros((9, 2)) # Position of the tubes
        # Keep track of the available tube coordinates
        free = np.ones((grid_size, grid_size))
        free[grid_size - tube_size+1:] = 0 # Can't fit a tube on the bottom edge of the volume
        free[:, grid_size - tube_size+1:] = 0 # Can't fit a tube on the right edge of the volume
        for ind_class in range(9):
            # Randomly draw a tube
            u = np.random.choice(grid_inds, size=1, replace=True, p=free.ravel()/np.sum(free)) # Randomly draw a tube coordinate
            x, y = grid_coord[u][0]
            free[max(x-tube_size+1, 0):x+tube_size, max(y-tube_size+1, 0):y+tube_size] = 0 # Update the available tube coordinates
            # Update generated dataset
            vol[x:x+tube_size, y:y+tube_size] = ind_class + 1
            rel_pos[x:x+tube_size, y:y+tube_size] = tube_coord
            corner[ind_class] = [x, y]
            # Check if there is still available tube coordinates, otherwise stop the loop
            if np.sum(free)==0:
                break
    # Grid rotation
    if rotation:
        rot, alpha, beta, gamma = get_rotation_matrix() # get_rot_mat(alpha, beta, gamma) #
        rot_vol_coord = np.linalg.inv(rot).dot(vol_coord.reshape(3, -1)).reshape(3, grid_size, grid_size, grid_size)
        my_interpolating_function = RegularGridInterpolator((xx, yy, zz), vol, bounds_error=False, fill_value=0, method='nearest')
        vol = my_interpolating_function(rot_vol_coord.reshape(3, -1).T)
        vol = vol.reshape(grid_size, grid_size, grid_size)
        my_interpolating_function = RegularGridInterpolator((xx, yy, zz), rel_pos, bounds_error=False, fill_value=0, method='nearest')
        rel_pos = my_interpolating_function(rot_vol_coord.reshape(3, -1).T)
        rel_pos = rel_pos.reshape(grid_size, grid_size, grid_size, 3)
    else:
        alpha, beta, gamma = 0, 0, 0
        rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if fixed_background:
        # Random label: each of the 8 tube gets a unique label between 1 and 9
        true_label = np.random.choice(np.arange(1, 10), size=9, replace=False)
    else:
        # Random label: each of the 8 tube gets a unique label between 0 and 9
        true_label = np.random.choice(np.arange(10), size=9, replace=False)
    for p in range(9):
        ground_truth_label[created_volume][vol==(p+1)] = true_label[p]
    if fixed_background:
        ground_truth_label[created_volume][vol==0] = 0
    else:
        # The background gets the label 10
        ground_truth_label[created_volume][vol==0] = 10

    ground_truth_relative_position[created_volume] = rel_pos
    ground_truth_grid_rotation[created_volume] = rot
    ground_truth_grid_rotation_angles[created_volume] = np.array([alpha, beta, gamma])
    ground_truth_top_left_corner[created_volume] = corner
    
    print(f'{created_volume/100 * dataset_size}%', end='\r')


# Save
os.makedirs(f'{dataset_path}/{prefix}_datasetsize_{dataset_size}_gridsize_{grid_size}_tubesize_{tube_size}_rotation_{rotation}_background_{fixed_background}/', exist_ok=True)

with gzip.open(f'{dataset_path}/{prefix}_datasetsize_{dataset_size}_gridsize_{grid_size}_tubesize_{tube_size}_rotation_{rotation}_background_{fixed_background}/ground_truth_label.pklz', 'wb') as f:
    pickle.dump(ground_truth_label, f)

with gzip.open(f'{dataset_path}/{prefix}_datasetsize_{dataset_size}_gridsize_{grid_size}_tubesize_{tube_size}_rotation_{rotation}_background_{fixed_background}/ground_truth_relative_position.pklz', 'wb') as f:
    pickle.dump(ground_truth_relative_position, f)

with gzip.open(f'{dataset_path}/{prefix}_datasetsize_{dataset_size}_gridsize_{grid_size}_tubesize_{tube_size}_rotation_{rotation}_background_{fixed_background}/ground_truth_grid_rotation.pklz', 'wb') as f:
    pickle.dump(ground_truth_grid_rotation, f)

with gzip.open(f'{dataset_path}/{prefix}_datasetsize_{dataset_size}_gridsize_{grid_size}_tubesize_{tube_size}_rotation_{rotation}_background_{fixed_background}/ground_truth_grid_rotation_angles.pklz', 'wb') as f:
    pickle.dump(ground_truth_grid_rotation_angles, f)

with gzip.open(f'{dataset_path}/{prefix}_datasetsize_{dataset_size}_gridsize_{grid_size}_tubesize_{tube_size}_rotation_{rotation}_background_{fixed_background}/ground_truth_top_left_corner.pklz', 'wb') as f:
    pickle.dump(ground_truth_top_left_corner, f)
