from torchvision import datasets
import torch
from utils import get_projection_grid, get_rotation_matrix, project_2d_on_sphere
import numpy as np
import gzip
import pickle
import os
import nibabel as nib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_path',
    required=True,
    help='Root path of the dataset (default: None)',
    type=str
)
parser.add_argument(
    '--mnist_path',
    required=True,
    help='path of the mnist dataset (default: None)',
    type=str
)
parser.add_argument(
    '--bandwidth',
    default=4,
    help='Bandwidth (default: 4)',
    type=int
)
parser.add_argument(
    '--rotation',
    default='no',
    help='Voxelwise rotation name (default: no)',
    choices=['no', 'voxel', 'same'],
    type=str
)
parser.add_argument(
    '--crop',
    default='no',
    help='Crop name (default: no)',
    choices=['no', 'local', 'random'],
    type=str
)
parser.add_argument(
    '--crop_size',
    default=14,
    help='Crop size (default: 14)',
    type=int
)
parser.add_argument(
    '--discard',
    action='store_true',
    help='Discard the image outside of the crop before the projection, otherwise keep the whole image and set to 0 outside of the crop (default: False)'
)
parser.add_argument(
    '--keep_no_crop',
    action='store_true',
    help='Keep the not crop spherical image (default: False)'
)
args = parser.parse_args()
dataset_path = args.dataset_path
mnist_path = args.mnist_path
bandwidth = args.bandwidth
crop = args.crop
crop_size = args.crop_size
discard = args.discard
keep_no_crop = args.keep_no_crop
rotation = args.rotation


mnist_data_folder = mnist_path + '/rawmnist/'
# Prepare MNIST dataset
print("getting mnist data")
trainset = datasets.MNIST(root=mnist_data_folder, train=True, download=True)
testset = datasets.MNIST(root=mnist_data_folder, train=False, download=True)
train_set, val_set = torch.utils.data.random_split(trainset, [50000, 10000], generator=torch.Generator().manual_seed(42))

mnist = {}
split_names = ['train', 'val', 'test']
split_N = np.zeros(len(split_names))
for i, split in enumerate(split_names):
    mnist[split] = {}
    if split=='train':
        mnist[split]['images'] = trainset.train_data.numpy()[train_set.indices]
        mnist[split]['labels'] = trainset.train_labels.numpy()[train_set.indices]
    elif split=='val':
        mnist[split]['images'] = trainset.train_data.numpy()[val_set.indices]
        mnist[split]['labels'] = trainset.train_labels.numpy()[val_set.indices]
    elif split=='test':
        mnist[split]['images'] = testset.test_data.numpy()
        mnist[split]['labels'] = testset.test_labels.numpy()
    split_N[i] = len(mnist[split]['labels'])

split_p = split_N / np.sum(split_N)

# Prepare spherical grid
grid = np.array(get_projection_grid(b=bandwidth))
nvec = grid[0].shape[0]

# Load ground truth
with gzip.open(f'{dataset_path}/ground_truth_label.pklz', 'rb') as f:
    ground_truth_label = pickle.load(f)
with gzip.open(f'{dataset_path}/ground_truth_relative_position.pklz', 'rb') as f:
    ground_truth_relative_position = pickle.load(f)
with gzip.open(f'{dataset_path}/ground_truth_grid_rotation.pklz', 'rb') as f:
    ground_truth_grid_rotation = pickle.load(f)
dataset_size, grid_size_x, grid_size_y, grid_size_z = ground_truth_label.shape
split_size = (dataset_size * split_p).astype(int)
split_size[0] += dataset_size - np.sum(split_size)
fixed_background = not 10 in np.unique(ground_truth_label)

offset = 0
for split, n_split in zip(split_names, split_size):
    data = mnist[split]
    signals = data['images'].reshape(-1, 28, 28).astype(np.float64)

    for i in range(n_split):
        print(f'split {split}: {i/n_split*100}%', end='\r')
        # select volume
        label = ground_truth_label[i + offset].astype(np.int32)
        rel_pos = ground_truth_relative_position[i + offset]
        signal_volume = np.zeros((grid_size_x, grid_size_y, grid_size_z, nvec))
        if keep_no_crop and crop!='no':
            image_no_crop = np.zeros((grid_size_x, grid_size_y, grid_size_z, nvec))
        # Select rotation matrix
        if rotation=='no':
            rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif rotation=='voxel':
            rot, alpha, beta, gamma = get_rotation_matrix()
        elif rotation=='same':
            rot = ground_truth_grid_rotation[i + offset]
        rotated_grid = np.linalg.inv(rot).dot(np.array(grid)) #rotate_grid(rot, grid)
        
        for j in range(11):
            if np.sum(label==j)!=0:
                if j!=10: # Each tube is made of random digits with same label as the tube
                    idxs = np.random.choice(np.arange(data['labels'].shape[0])[data['labels']==j], size=np.sum(label==j))
                else: # Background is made of random digits from all classes
                    idxs = np.random.choice(np.arange(data['labels'].shape[0]), size=np.sum(label==j))
                if crop=='no': # Use the full digit (not crop)
                    chunk = signals[idxs]
                else: # Use crop
                    if crop=='local' and (j!=10 or (j==0 and fixed_background)): # Coordinate based crop
                        a = rel_pos[label==j]
                        x = np.zeros(idxs.shape[0]).astype(int)
                        y = np.zeros(idxs.shape[0]).astype(int)
                        step = (28-crop_size)/(a[:,0].max()-1)
                        for k in range(a[:,0].max()):
                            x[a[:, 0] == k+1] = int(step*k)
                            y[a[:, 1] == k+1] = int(step*k)
                    elif crop=='random' or j==10 or (j==0 and fixed_background): # random crop, background is always random crop
                        x = np.random.randint(0, 28 - crop_size, size=idxs.shape[0])
                        y = np.random.randint(0, 28 - crop_size, size=idxs.shape[0])
                    if not discard: # Don't discard the whole image, set to 0 outside of the crop
                        chunk = signals[idxs]
                    else: # Discard the image out of crop
                        chunk = np.zeros((len(idxs), crop_size, crop_size))
                    for z in range(len(idxs)):
                        if not discard:
                            chunk[z, :x[z]] = 0
                            chunk[z, :, :y[z]] = 0
                            chunk[z, x[z]+crop_size:] = 0
                            chunk[z, :, y[z]+crop_size:] = 0
                        else:
                            chunk[z] = signals[idxs[z], x[z]:x[z]+crop_size, y[z]:y[z]+crop_size]

                signal_volume[label==j] = project_2d_on_sphere(chunk, rotated_grid)
                if keep_no_crop and crop!='no':
                    chunk_nocrop = signals[idxs]
                    image_no_crop[label==j] = project_2d_on_sphere(chunk_nocrop, rotated_grid)

        
        signal_volume = signal_volume / 255
        os.makedirs(f'{dataset_path}/bandwidth_{bandwidth}_rotation_{rotation}_crop_{crop}_cropsize_{crop_size}_discard_{discard}_keepnocrop_{keep_no_crop}/{split}/sub_{i + offset}', exist_ok=True)
        clipped_img = nib.Nifti1Image(signal_volume, np.eye(4))
        nib.save(clipped_img, f'{dataset_path}/bandwidth_{bandwidth}_rotation_{rotation}_crop_{crop}_cropsize_{crop_size}_discard_{discard}_keepnocrop_{keep_no_crop}/{split}/sub_{i + offset}/image.nii.gz')
        if keep_no_crop and crop!='no':
            image_no_crop = image_no_crop / 255
            clipped_img = nib.Nifti1Image(image_no_crop, np.eye(4))
            nib.save(clipped_img, f'{dataset_path}/bandwidth_{bandwidth}_rotation_{rotation}_crop_{crop}_cropsize_{crop_size}_discard_{discard}_keepnocrop_{keep_no_crop}/{split}/sub_{i + offset}/image_no_crop.nii.gz')
        clipped_img = nib.Nifti1Image(label, np.eye(4))
        nib.save(clipped_img, f'{dataset_path}/bandwidth_{bandwidth}_rotation_{rotation}_crop_{crop}_cropsize_{crop_size}_discard_{discard}_keepnocrop_{keep_no_crop}/{split}/sub_{i + offset}/label.nii.gz')
    offset += n_split
