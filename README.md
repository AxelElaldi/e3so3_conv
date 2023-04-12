# Roto-Translation Equivariant Spherical Deconvolution
This repo contains the PyTorch implementation of [E3 x SO3 Equivariant Networks for Spherical Deconvolution in Diffusion MRI](https://openreview.net/pdf?id=lri_iAbpn_r). The main application pertains to fODF estimation in diffusion MRI, however it extends to generic learning problems on a structured or unstructured spatial configuration of spherical measurements.

We provide code for both generic usage and to perform the R3 x S2 MNIST and diffusion MRI experiments from the paper.

<p align="center">
  <img src="https://github.com/AxelElaldi/e3so3_conv/blob/main/img/overview.png" />
</p>
Figure: (A) Diffusion MRI measures a spatial grid of spherical signals. (B) In addition to translations and grid reflections, we construct layers equivariant to voxel and grid-wise rotations and any combination thereof. (C) RT-ESD takes a patch of spheres and processes it with an E(3) x SO(3)-equivariant UNet to produce fODFs. It is trained under an unsupervised regularized reconstruction objective.

## 1. Getting started

Set up the python environment:

```
conda create -n rtesd python=3.8
source activate rtesd
pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP
pip install numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install nibabel
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # Tested for PyTorch 1.10
pip install healpy
pip install tensorboard
pip install wandb # If you want to use wandb
pip install torchio # We use torchio to work on the proposed MNIST dataset
```

We use [WandB](https://docs.wandb.ai/quickstart) to record model training. This is optional and disabled by default.

## 2. Layer overview

<p align="center">
  <img src="https://github.com/AxelElaldi/e3so3_conv/blob/main/img/e3so3conv.png" />
</p>

(a) The input is a patch of spherical signals $\mathbf{f}$ with $F_{in}$ features. For each voxel $x\in\mathbb{R}^3$, $\mathbf{f}(x)$ is projected onto a spherical graph $\mathcal{G}$ with $V$ nodes. (b) The convolution first filters each sphere with Chebyshev polynomials applied to the Laplacian $L$. The filter outputs are then aggregated along the grid to create a spherical signal $\mathbf{\hat{f}}$ with $F_{in}V$ features. (c) For each $v\in\mathcal{G}$, we extract the corresponding spatial signal $\mathbf{\hat{f}}_v(.)$. (d) These $V$ convolutions give the final grid of spheres $\mathbf{f}_{out}$. Connected boxes across (c) and (d) show spatial operations on a single spherical vertex.

We use the spherical graph convolution from [DeepSphere](https://github.com/deepsphere/deepsphere-pytorch) and the base code from [ESD](https://github.com/AxelElaldi/equivariant-spherical-deconvolution).


## 3. E(3) x SO(3) convolution example
```python:
from model.graphconv import Conv
from utils.sampling import HealpixSampling
import torch
```

We first need to define the spherical sampling used by the convolution. It will create the laplacian of the spherical graph and, the spherical pooling operations, which is dependent on the spherical graph. We combine all these information into one SphericalSampling class.

```python:
# Define the spherical sampling used for the spherical convolution.
# The Healpix sampling is convenient thanks to its hierarchical structures
# making it easier to define the pooling operation.
n_side = 8
depth = 2 # Number of hierarchical level to define for this sampling. It will automatically create the pooling operations.
patch_size = 5 # Spatial size of the sampling. Used to efficiently create the pooling operations.
sh_degree = 8
pooling_mode = 'average' # Choice between average and max pooling
pooling_name = 'mixed' # Choice between spatial, spherical, or a mixed of both.
sampling = HealpixSampling(n_side, depth, patch_size, sh_degree, pooling_mode, pooling_name)

# Access the laplacians and pooling of the sampling
laps = sampling.laps # List of the laplacians from lowest to highest resolution sampling
pools = sampling.pooling # List of the poolings from lowest to highest resolution sampling
```

Once we know the spherical graph laplacian, we create our convolution operator. In addition to our E3xSO3 convolution, we provides the more usual spatial and spherical convolutions.
```python:
# Define convolution layers for the highest resolution sampling we previously defined
in_channels = 2 # Number of input channel
out_channels = 2 # Number of output channel

kernel_sizeSph = 3 # Spherical kernel size
kernel_sizeSpa = 3 # Spatial kernel size

lap = laps[-1] # Laplacian of the spherical graph
bias = True # Add bias after convolution

# We implemented three convolutions
conv_name = 'spherical'
so3_conv = Conv(in_channels, out_channels, lap, kernel_sizeSph, kernel_sizeSpa, bias, conv_name)

conv_name = 'spatial' # being e3 equivariant because of isotropic filters
isoSpa = True # Use isotropic filter for the spatial convolution
e3_conv = Conv(in_channels, out_channels, lap, kernel_sizeSph, kernel_sizeSpa, bias, conv_name, isoSpa)

conv_name = 'mixed'
isoSpa = True # Use isotropic filter for the spatial convolution
e3so3_conv = Conv(in_channels, out_channels, lap, kernel_sizeSph, kernel_sizeSpa, bias, conv_name, isoSpa)
```

These three convolutions can readily be applied to a R3xS2 random signal.
```python:
# Generate a random R3 x S2 signal
batch_size = 2
# Convolution input should have size
# Batch x Feature Channel x Number of spherical vertice x Spatial patch size x Spatial patch size x Spatial patch size
x = torch.rand(batch_size, in_channels, lap.shape[0], patch_size, patch_size, patch_size)

# The spherical convolution is only applied to the spherical vertices
# without knowledge of the spatial neighborhoods
out_so3 = so3_conv(x)

# The spatial convolution is only applied to the spatial dimension
# considering each spherical vertex as 3D features
out_e3 = e3_conv(x)

# The mixed convolution is a mixed between the spherical and spatial convolutions,
# leveraging the spherical and the spatial structures.
out_e3so3 = e3so3_conv(x)
```

## 4. E(3) x SO(3) Unet example
```python:
from model.unet import GraphCNNUnet
from utils.sampling import HealpixSampling
import torch
```

Again, we first define the spherical sampling.
```python:
# Define the spherical sampling used for the spherical convolution.
# The Healpix sampling is convenient thanks to its hierarchical structures
# making it easier to define the pooling operation.
n_side = 8
depth = 2 # Number of hierarchical level to define for this sampling. It will automatically create the pooling operations.
patch_size = 5 # Spatial size of the sampling. Used to efficiently create the pooling operations.
sh_degree = 8
pooling_mode = 'average' # Choice between average and max pooling
pooling_name = 'mixed' # Choice between spatial, spherical, or a mixed of both.
sampling = HealpixSampling(n_side, depth, patch_size, sh_degree, pooling_mode, pooling_name)

# Access the laplacians and pooling of the sampling
laps = sampling.laps # List of the laplacians from lowest to highest resolution sampling
pools = sampling.pooling # List of the poolings from lowest to highest resolution sampling
vecs = sampling.vec
```

We then create the UNet. Again, in addition to the proposed E3xSO3 convolution, we provide implementation for the spatial and spherical UNet.
```python:
in_channels = 1 # Number of input channel
out_channels = 5 # Number of output channel
filter_start = 1 # Number of filters after the first convolution. Then, number of filter double after each pooling
block_depth = 1 # Number de block(convolution + bn + activation per) between two poolings for encoder
in_depth = 1 # Number de block(convolution + bn + activation per) before unpooling for decoder
kernel_sizeSph = 3 # Spherical kernel size
kernel_sizeSpa = 3 # Spatial kernel size
poolings = pools # List of poolings
laps = laps # List of laplacians
conv_name = pooling_name # Name of the convolution
isoSpa = True # Use istropic spatial filter to get E3 equivariance
keepSphericalDim = True # For output, keep the spherical dimension or global average across vertices
vec = vecs # list of vertex coordinates, for the Muller convolution.

unet = GraphCNNUnet(in_channels, out_channels, filter_start, block_depth, in_depth, kernel_sizeSph, kernel_sizeSpa, poolings, laps, conv_name, isoSpa, keepSphericalDim, vec)

# Generate a random R3xS2 signal
batch_size = 1
# Convolution input should have size
# Batch x Feature Channel x Number of spherical vertice x Spatial patch size x Spatial patch size x Spatial patch size
x = torch.rand(batch_size, in_channels, laps[-1].shape[0], patch_size, patch_size, patch_size) # B x F_in x V x P x P x P

y = unet(x) # B x F_out x (V or 1) x P x P x P
```

## 5. R3 x S2 MNIST dataset

We provide the code to generate different versions of the R3xS2 MNIST dataset. For more details on the generation process, we refer to the paper [RTESD](https://openreview.net/pdf?id=lri_iAbpn_r).

<p align="center">
  <img src="https://github.com/AxelElaldi/e3so3_conv/blob/main/img/r3s2mnist.png" />
</p>

Spatio-spherical images and label maps for $\mathbb{R}^3 \times \mathcal{S}^2$ MNIST, respectively.

### 5.1 Create the volume labels
We first create the spatial volumes. The snapshot bellow create 100 volumes (dataset_size) of size 16x16x16 (grid_size). A tube of size 4x4x16 (tube_size) is created for each digit. Each volume is randomly rotated and the background is fixed to the digit 0. You need to choose the root path of the dataset. 
```
python create_volume.py --dataset_path $your_dataset_path --prefix mnist --dataset_size 100 --grid_size 16 --tube_size 4 --rotation --fixed_background
```

### 5.2 Create the spherical signals
We then create the spherical images for each voxel of each volumes. The snapshot bellow takes the previous 100 volumes. For each voxel, we sample a random image from the mnist dataset from the corresponding digit class and take a random crop (crop) of size 14x14 (crop_size). The crop image is then projected onto a Healpix grid of bandwidth 4, i.e. 192 vertices.
```
python create_sphere.py --dataset_path $your_dataset_path/mnist_datasetsize_100_gridsize_16_tubesize_4_rotation_True_background_True --mnist_path $your_dataset_path --bandwidth 4 --rotation voxel --crop random --crop_size 14 --discard --keep_no_crop
```

You will find in your dataset path the resulting dataset. It is split between train/val/test. Each generated volume has an image and label files saved under a nii.gz format (nifti). We also provide the image with the full spherical digit under the file image_np_crop.nii.gz


## 6. R3xS2 MNIST classification

### 6.1 Dataset generation
We provide the training and testing script used in our paper. First, generate the five datasets.
```
% No rotation
python create_volume.py --dataset_path $your_dataset_path --prefix norotcrop --dataset_size 100 --grid_size 16 --tube_size 4  --fixed_background
python create_sphere.py --dataset_path $your_dataset_path/norotcrop_datasetsize_100_gridsize_16_tubesize_4_rotation_False_background_True --mnist_path $your_dataset_path --bandwidth 4 --rotation no --crop random --crop_size 14 --discard --keep_no_crop

% Voxel rotation
python create_volume.py --dataset_path $your_dataset_path --prefix voxelcrop --dataset_size 100 --grid_size 16 --tube_size 4  --fixed_background
python create_sphere.py --dataset_path $your_dataset_path/voxelcrop_datasetsize_100_gridsize_16_tubesize_4_rotation_False_background_True --mnist_path $your_dataset_path --bandwidth 4 --rotation voxel --crop random --crop_size 14 --discard --keep_no_crop

% Grid rotation
python create_volume.py --dataset_path $your_dataset_path --prefix gridcrop --dataset_size 100 --grid_size 16 --tube_size 4 --rotation --fixed_background
python create_sphere.py --dataset_path $your_dataset_path/gridcrop_datasetsize_100_gridsize_16_tubesize_4_rotation_True_background_True --mnist_path $your_dataset_path --bandwidth 4 --rotation no --crop random --crop_size 14 --discard --keep_no_crop

% Grid and voxel rotation
python create_volume.py --dataset_path $your_dataset_path --prefix gridvoxelcrop --dataset_size 100 --grid_size 16 --tube_size 4 --rotation --fixed_background
python create_sphere.py --dataset_path $your_dataset_path/gridvoxelcrop_datasetsize_100_gridsize_16_tubesize_4_rotation_True_background_True --mnist_path $your_dataset_path --bandwidth 4 --rotation voxel --crop random --crop_size 14 --discard --keep_no_crop

% Same grid and voxel rotation
python create_volume.py --dataset_path $your_dataset_path --prefix gridvoxelsamecrop --dataset_size 100 --grid_size 16 --tube_size 4 --rotation --fixed_background
python create_sphere.py --dataset_path $your_dataset_path/gridvoxelsamecrop_datasetsize_100_gridsize_16_tubesize_4_rotation_True_background_True --mnist_path $your_dataset_path --bandwidth 4 --rotation same --crop random --crop_size 14 --discard --keep_no_crop
```

### 6.2 Model training
Train the model with the following command:
```
python train_mnist.py --data_path $your_dataset_path --sfx_train norot --batch_size 32 --lr 1e-2 --epoch 50 --conv_name mixed --kernel_sizeSph 3 --kernel_sizeSpa 3  --depth 3 --start_filter 8 --bandwidth 4 --save_every 1 --cropsize 14 --crop random --background
```

You can train on different dataset version using sfx_train --> [norot, voxel, grid, gridvoxel, gridvoxelsame]. You can use different convolution using conv_name --> [mixed, spherical, spatial_vec, spatial_sh, spatial]

### 6.3 Model testing
Test the model with the following command:
```
python test_mnist.py --data_path $your_dataset_path --batch_size 1 --model_name $your_model_name --epoch $your_model_epoch
```

### 6.4 Result

<p align="center">
  <img src="https://github.com/AxelElaldi/e3so3_conv/blob/main/img/r3s2mnist_result.png" />
</p>

Classification performances when trained on data with (right) or without (left) rotation augmentation and tested on data with no rotations, grid-rotations, voxel-rotations, and independent grid and voxel-rotations.

## 7. Diffusion MRI deconvolution
The main application of this work is for dMRI deconvolution. We use the same architecture and training process as [ESD](https://github.com/AxelElaldi/equivariant-spherical-deconvolution), where you can find usefull information on the deconvolution architecture.

<p align="center">
  <img src="https://github.com/AxelElaldi/e3so3_conv/blob/main/img/unet_dmri.png" />
</p>

RT-ESD takes a patch of spheres and processes it with an $E(3)\times SO(3)$-equivariant UNet to produce fODFs. It is trained under an unsupervised regularized reconstruction objective.

## 7.1 Prepare the diffusion MRI data

In a root folder:
* Copy your diffusion MRI data (resp. the mask) as a nifti file under the name **features.nii.gz** (**mask.nii.gz**). 
* Copy your bvecs and bvals files under the names **bvecs.bvecs** and **bvals.bvals**.
* In the root folder, create a folder for the response functions, called **response_functions**. There, create a folder for each response function estimation algorithm you want to use. We will use the name **rf_algo** as example folder. In each algorithm folder, copy the white matter, grey matter, and CSF reponse function files under the names **wm_response.txt**, **gm_response.txt**, and **csf_response.txt**. We refer to [Mrtrix3](https://mrtrix.readthedocs.io/en/0.3.16/concepts/response_function_estimation.html) for different response function algorithms.


## 7.2 Train a model
You can train a new model on your data using the following bash command:

```
    python train.py --data_path $your_data_path --batch_size 32 --lr 0.0017 --epoch 50 --filter_start 1 --sh_degree 18 --save_every 1 --loss_intensity L2 --loss_sparsity cauchy --loss_non_negativity L2 --sigma_sparsity 1e-05 --sparsity_weight 1e-06 --intensity_weight 1 --nn_fodf_weight 1 --pve_weight 1e-11 --wm --gm  --rf_name $your_rf_algo_choice --depth 5 --patch_size 5 --normalize --kernel_sizeSpa 3 --conv_name mixed
```
Training script works with mixed (RT-ESD) and spherical (ESD) convolutions. Adding the --concatenate flag produces a (C-ESD) model.

## 7.3 Test a model
You can test a trained model on your data using the following bash command:

```
    python test.py --data_path $your_data_path --batch_size 1 --epoch $your_model_epoch --model_name $your_model_name --middle_voxel
```

## 7.4 Result
<p align="center">
  <img src="https://github.com/AxelElaldi/e3so3_conv/blob/main/img/pve.png" />
</p>

Unsupervised partial volume estimation. Col. 1: T1w MRI and label map of the subject co-registered to the dMRI input. Cols. 2--4, row 1: Partial volume estimates from each deconvolution method. Cols. 2--4, row 2: Divergence maps between the estimated partial volumes and the reference segmentation.

<p align="center">
  <img src="https://github.com/AxelElaldi/e3so3_conv/blob/main/img/fiber.png" />
</p>

Estimated fODFs from the Tractometer dMRI dataset. This figure visualizes results from CSD, ESD, and RT-ESD at a particular location with crossing fibers. RT-ESD yields more spatially-coherent fiber directions with accurate modeling of crossing fibers as compared to the spatially-agnostic ESD and CSD baselines.

## Acknowledgments

Part of the graph convolution code comes from [DeepSphere](https://github.com/deepsphere/deepsphere-pytorch).

Please consider citing our paper if you find this repository useful.
```
@inproceedings{
elaldi2023e,
title={E(3) x {SO}(3)-Equivariant Networks for Spherical Deconvolution in Diffusion {MRI}},
author={Axel Elaldi and Guido Gerig and Neel Dey},
booktitle={Medical Imaging with Deep Learning},
year={2023},
url={https://openreview.net/forum?id=lri_iAbpn_r}
}
```
