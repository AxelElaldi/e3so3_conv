from torch.utils.data import Dataset
import nibabel as nib
import torch
import os
import numpy as np
import torchio as tio


class DMRIDataset(Dataset):
    def __init__(self, data_path, mask_path, bvec_path, bval_path, patch_size, test=False, concatenate=False):
        self.patch_size = patch_size
        try:
            self.data = nib.load(data_path)
        except:
            self.data = nib.load(data_path+'.gz')
        self.affine = self.data.affine
        self.header = self.data.header
        self.data = torch.Tensor(self.data.get_fdata()) # Load image X x Y x Z x V
        # Load mask
        if os.path.isfile(mask_path) or os.path.isfile(mask_path+'.gz'):
            try:
                self.mask = torch.Tensor(nib.load(mask_path).get_fdata()) # X x Y x Z
            except:
                self.mask =  torch.Tensor(nib.load(mask_path+'.gz').get_fdata())
        else:
            self.mask = torch.ones(self.data.shape[:-1])
        # 0-pad image and mask
        self.data = torch.nn.functional.pad(self.data, (0, 0, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2), 'constant', value=0)
        self.mask = torch.nn.functional.pad(self.mask, (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2), 'constant', value=0)
        # Permute
        self.data = self.data.permute(3, 0, 1, 2)
        # Save the non-null index of the mask
        ind = np.arange(self.mask.nelement())[torch.flatten(self.mask) != 0]
        self.x, self.y, self.z = np.unravel_index(ind, self.mask.shape)
        self.N = len(self.x)
        print('Dataset size: {}'.format(self.N))
        print('Padded scan size: {}'.format(self.data.shape))
        print('Padded mask size: {}'.format(self.mask.shape))
        vectors = np.loadtxt(bvec_path)
        shell = np.loadtxt(bval_path)
        if vectors.shape[0] == 3:
            vectors = vectors.T
        assert shell.shape[0] == vectors.shape[0]
        assert vectors.shape[1] == 3
        vectors[:, 0] = -vectors[:, 0] # (should use the affine tranformation matrix instead)
        self.vec = np.unique(vectors[shell!=0], axis=0)
        self.b = np.sort(np.unique(shell))
        self.normalization_value = np.ones(len(self.b))
        self.patch_size_output = patch_size
        self.patch_size_input = patch_size
        self.concatenate = concatenate
        if concatenate:
            self.patch_size_output = 1


    def __len__(self):
        return int(self.N)

    def __getitem__(self, i):
        input = self.data[None, :, self.x[i] - (self.patch_size // 2):self.x[i] + (self.patch_size // 2) + (self.patch_size%2), self.y[i] - (self.patch_size // 2):self.y[i] + (self.patch_size // 2) + (self.patch_size%2), self.z[i] - (self.patch_size // 2):self.z[i] + (self.patch_size // 2) + (self.patch_size%2)] # 1 x V x P x P x P
        if self.concatenate:
            input = torch.flatten(input[0], start_dim=-3) # V x P*P*P
            input = input.permute(1, 0)[:, :, None, None, None] #  P*P*P x V x 1 x 1 x 1
        output = self.data[:, self.x[i] - (self.patch_size_output // 2):self.x[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.y[i] - (self.patch_size_output // 2):self.y[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.z[i] - (self.patch_size_output // 2):self.z[i] + (self.patch_size_output // 2) + (self.patch_size_output%2)] # V x P x P x P
        mask = self.mask[self.x[i] - (self.patch_size_output // 2):self.x[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.y[i] - (self.patch_size_output // 2):self.y[i] + (self.patch_size_output // 2) + (self.patch_size_output%2), self.z[i] - (self.patch_size_output // 2):self.z[i] + (self.patch_size_output // 2) + (self.patch_size_output%2)] # P x P x P
        return {'sample_id': i, 'input': input, 'out': output, 'mask': mask}


def mnist_dataset(path, sfx, bandwidth, split, centered=True, crop='local', cropsize=14, background=False):
    subjects_list = []
    if sfx=='grid' or sfx=='gridvoxel' or sfx=='gridvoxelsame':
        grid_rotation = True
    else:
        grid_rotation = False
    if sfx=='voxel' or sfx=='gridvoxel':
        voxel_rotation = 'voxel'
    elif sfx=='gridvoxelsame':
        voxel_rotation = 'same'
    else:
        voxel_rotation = 'no'
    path_image = f"{path}/{sfx}{centered*'crop'}_datasetsize_100_gridsize_16_tubesize_4_rotation_{grid_rotation}_background_{background}/bandwidth_{bandwidth}_rotation_{voxel_rotation}_crop_{crop}_cropsize_{cropsize}_discard_{centered}_keepnocrop_True/{split}/"
    sub_name = os.listdir(path_image)
    for sub in sub_name: 
        subject = tio.Subject(
            image=tio.ScalarImage(f'{path_image}/{sub}/image.nii.gz'),
            reconstruction=tio.ScalarImage(f'{path_image}/{sub}/image_no_crop.nii.gz'),
            label=tio.LabelMap(f'{path_image}/{sub}/label.nii.gz'),
            name=sub
        )
        subjects_list.append(subject)
    
    subjects_dataset = tio.SubjectsDataset(subjects_list)
    return subjects_dataset
