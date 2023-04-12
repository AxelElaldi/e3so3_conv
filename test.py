import argparse
import os
import numpy as np
import nibabel as nib
import json

from utils.sampling import HealpixSampling, ShellSampling, BvecSampling
from utils.dataset import DMRIDataset
from utils.response import load_response_function
from model.model import Model

import torch
from torch.utils.data.dataloader import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, batch_size, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         normalize, model_name, epoch, patch_size, middle_voxel, graph_sampling, conv_name, isoSpa, concatenate):
    """Test a model
    Args:
        data_path (str): Data path
        batch_size (int): Batch size
        n_epoch (int): Number of training epoch
        kernel_size (int): Kernel Size
        filter_start (int): Number of output features of the first convolution layer
        sh_degree (int): Spherical harmonic degree of the fODF
        depth (int): Graph subsample depth
        n_side (int): Resolution of the Healpix map
        rf_name (str): Response function algorithm name
        wm (float): Use white matter
        gm (float): Use gray matter
        csf (float): Use CSF
        normalize (bool): Normalize the fODFs
        load_state (str): Load pre trained network
        model_name (str): Name of the model folder
        epoch (int): Epoch to use for testing
    """
    # Load the shell and the graph samplings
    dataset = DMRIDataset(f'{data_path}/features.nii', f'{data_path}/mask.nii', f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', patch_size, test=True, concatenate=concatenate)
    bs = patch_size//2
    feature_in = 1
    if concatenate:
        feature_in = patch_size*patch_size*patch_size
        patch_size = 1
    bvec = dataset.vec
    if graph_sampling=='healpix':
        graphSampling = HealpixSampling(n_side, depth, patch_size, sh_degree=sh_degree, pooling_name=conv_name)
    elif graph_sampling=='bvec':
        graphSampling = BvecSampling(bvec, depth, image_size=patch_size, sh_degree=sh_degree, pooling_mode='average')
    else:
        raise NotImplementedError
    shellSampling = ShellSampling(f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', sh_degree=sh_degree, max_sh_degree=8)

    # Load the image and the mask
    dataloader_test = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    n_batch = len(dataloader_test)
    
    # Load the Polar filter used for the deconvolution
    polar_filter_equi, polar_filter_inva = load_response_function(f'{data_path}/response_functions/{rf_name}', wm=wm, gm=gm, csf=csf, max_degree=sh_degree, n_shell=len(shellSampling.shell_values), norm=dataset.normalization_value)

    # Create the deconvolution model and load the trained model
    model = Model(polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in)
    model.load_state_dict(torch.load(f'{data_path}/result/{model_name}/history/epoch_{epoch}.pth'), strict=False)
    # Load model in GPU
    model = model.to(DEVICE)
    model.eval()

    # Output initialization
    if middle_voxel:
        b_selected = 1
        b_start = patch_size//2
        b_end = b_start + 1
    else:
        b_selected = patch_size
        b_start = 0
        b_end = b_selected

    nb_coef = int((sh_degree + 1) * (sh_degree / 2 + 1))
    count = np.zeros((dataset.data.shape[1],
                    dataset.data.shape[2],
                    dataset.data.shape[3]))
    reconstruction_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], len(shellSampling.vectors)))
    if wm:
        fodf_shc_wm_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], nb_coef))
    if gm:
        fodf_shc_gm_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], 1))
    if csf:
        fodf_shc_csf_list = np.zeros((dataset.data.shape[1],
                                    dataset.data.shape[2],
                                    dataset.data.shape[3], 1))
    # Test on batch.
    for i, data in enumerate(dataloader_test):
        print(str(i * 100 / n_batch) + " %", end='\r', flush=True)
        # Load the data in the DEVICE
        input = data['input'].to(DEVICE)
        sample_id = data['sample_id']

        x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(input)
        
        for j in range(len(input)):
            sample_id_j = sample_id[j]
            reconstruction_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_reconstructed[j, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            if wm:
                fodf_shc_wm_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_deconvolved_equi_shc[j, 0, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            index = 0
            if gm:
                fodf_shc_gm_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                index += 1
            if csf:
                fodf_shc_csf_list[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += x_deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            count[dataset.x[sample_id_j] - (b_selected // 2):dataset.x[sample_id_j] + (b_selected // 2) + (b_selected%2),
                dataset.y[sample_id_j] - (b_selected // 2):dataset.y[sample_id_j] + (b_selected // 2) + (b_selected%2),
                dataset.z[sample_id_j] - (b_selected // 2):dataset.z[sample_id_j] + (b_selected // 2) + (b_selected%2)] += 1
        '''
        for j in range(len(input)):
            sample_id_j = sample_id[j]
            print(dataset.x[sample_id_j], dataset.y[sample_id_j], dataset.z[sample_id_j])
            reconstruction_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_reconstructed[j, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            if wm:
                fodf_shc_wm_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_deconvolved_equi_shc[j, 0, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            index = 0
            if gm:
                fodf_shc_gm_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                index += 1
            if csf:
                fodf_shc_csf_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
            count[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += 1
            print(np.sum(count[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected]))
        '''
    # Average patch
    try:
        reconstruction_list[count!=0] = reconstruction_list[count!=0] / count[count!=0, None]
        if wm:
            fodf_shc_wm_list[count!=0] = fodf_shc_wm_list[count!=0] / count[count!=0, None]
        if gm:
            fodf_shc_gm_list[count!=0] = fodf_shc_gm_list[count!=0] / count[count!=0, None]
        if csf:
            fodf_shc_csf_list[count!=0] = fodf_shc_csf_list[count!=0] / count[count!=0, None]
    except:
        print('Count failed')
    
    # Save the results
    #bs = 0
    if bs>0:
        count = count[bs:-bs,bs:-bs,bs:-bs]
    #else:
    #    count = count[:dataset.X,:dataset.Y,:dataset.Z]
    count = np.array(count).astype(np.float32)
    img = nib.Nifti1Image(count, dataset.affine, dataset.header)
    nib.save(img, f"{data_path}/result/{model_name}/test{'_middle'*middle_voxel}/epoch_{epoch}/count.nii.gz")
    if bs>0:
        reconstruction_list = reconstruction_list[bs:-bs,bs:-bs,bs:-bs]
    #else:
    #    reconstruction_list = reconstruction_list[:dataset.X,:dataset.Y,:dataset.Z]
    reconstruction_list = np.array(reconstruction_list).astype(np.float32)
    img = nib.Nifti1Image(reconstruction_list, dataset.affine, dataset.header)
    nib.save(img, f"{data_path}/result/{model_name}/test{'_middle'*middle_voxel}/epoch_{epoch}/reconstruction.nii.gz")
    if wm:
        if bs>0:
            fodf_shc_wm_list = fodf_shc_wm_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_wm_list = fodf_shc_wm_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_wm_list = np.array(fodf_shc_wm_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_wm_list, dataset.affine, dataset.header)
        nib.save(img, f"{data_path}/result/{model_name}/test{'_middle'*middle_voxel}/epoch_{epoch}/fodf.nii.gz")
    if gm:
        if bs>0:
            fodf_shc_gm_list = fodf_shc_gm_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_gm_list = fodf_shc_gm_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_gm_list = np.array(fodf_shc_gm_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_gm_list, dataset.affine, dataset.header)
        nib.save(img, f"{data_path}/result/{model_name}/test{'_middle'*middle_voxel}/epoch_{epoch}/fodf_gm.nii.gz")
    if csf:
        if bs>0:
            fodf_shc_csf_list = fodf_shc_csf_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_csf_list = fodf_shc_csf_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_csf_list = np.array(fodf_shc_csf_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_csf_list, dataset.affine, dataset.header)
        nib.save(img, f"{data_path}/result/{model_name}/test{'_middle'*middle_voxel}/epoch_{epoch}/fodf_csf.nii.gz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        required=True,
        help='Root path of the data (default: None)',
        type=str
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        help='Batch size (default: 32)',
        type=int
    )
    parser.add_argument(
        '--model_name',
        required=True,
        help='Epoch (default: None)',
        type=str
    )
    parser.add_argument(
        '--epoch',
        required=True,
        help='Epoch (default: None)',
        type=int
    )
    parser.add_argument(
        '--middle_voxel',
        action='store_true',
        help='Extract only middle voxel of a patch (default: False)',
    )
    args = parser.parse_args()
    # Test properties
    batch_size = args.batch_size
    middle_voxel = args.middle_voxel
    
    # Data path
    data_path = args.data_path
    assert os.path.exists(data_path)

    # Load trained model
    model_name = args.model_name
    assert os.path.exists(f'{data_path}/result/{model_name}')
    epoch = args.epoch
    assert os.path.exists(f'{data_path}/result/{model_name}/history/epoch_{epoch}.pth')

    # Load parameters
    with open(f'{data_path}/result/{model_name}/args.txt', 'r') as file:
        args_json = json.load(file)
    
    # Model architecture properties
    filter_start = int(args_json['filter_start'])
    sh_degree = int(args_json['sh_degree'])
    kernel_sizeSph = int(args_json['kernel_sizeSph'])
    kernel_sizeSpa = int(args_json['kernel_sizeSpa'])
    depth = int(args_json['depth'])
    n_side = int(args_json['n_side'])
    normalize = bool(args_json['normalize'])
    patch_size = int(args_json['patch_size'])
    try:
        graph_sampling = int(args_json['graph_sampling'])
    except:
        graph_sampling = 'healpix'
    conv_name = str(args_json['conv_name']) 
    isoSpa = not bool(args_json['anisoSpa']) 
    patch_size = int(args_json['patch_size'])
    try:
        concatenate = bool(args_json['concatenate'])
    except:
        concatenate = False

    # Load response functions
    rf_name = str(args_json['rf_name'])
    wm = bool(args_json['wm'])
    gm = bool(args_json['gm'])
    csf = bool(args_json['csf'])

    print(f'Filter start: {filter_start}')
    print(f'SH degree: {sh_degree}')
    print(f'Kernel size spherical: {kernel_sizeSph}')
    print(f'Kernel size spatial: {kernel_sizeSpa}')
    print(f'Unet depth: {depth}')
    print(f'N sidet: {n_side}')
    print(f'fODF normalization: {normalize}')
    print(f'Patch size: {patch_size}')
    print(f'RF name: {rf_name}')
    print(f'Use WM: {wm}')
    print(f'Use GM: {gm}')
    print(f'Use CSF: {csf}')
    print(f'Concatenate: {concatenate}')

    # Test directory
    test_path = f"{data_path}/result/{model_name}/test{'_middle'*middle_voxel}/epoch_{epoch}"
    if not os.path.exists(test_path):
        print('Create new directory: {0}'.format(test_path))
        os.makedirs(test_path)


    main(data_path, batch_size, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         normalize, model_name, epoch, patch_size, middle_voxel, graph_sampling, conv_name, isoSpa, concatenate)

