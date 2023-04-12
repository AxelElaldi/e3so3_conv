import argparse
import os
import numpy as np
import pickle
import json
import time

from utils.loss import Loss
from utils.sampling import HealpixSampling, ShellSampling, BvecSampling
from utils.dataset import DMRIDataset
from model.shutils import ComputeSignal
from utils.response import load_response_function
from model.model import Model
use_wandb = False
if use_wandb:
    import wandb

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, batch_size, lr, n_epoch, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         loss_fn_intensity, loss_fn_non_negativity, loss_fn_sparsity, sigma_sparsity,
         intensity_weight, nn_fodf_weight, sparsity_weight, pve_weight,
         save_path, save_every, normalize, load_state, patch_size, graph_sampling, conv_name, isoSpa, concatenate, middle_voxel):
    """Train a model
    Args:
        data_path (str): Data path
        batch_size (int): Batch size
        lr (float): Learning rate
        n_epoch (int): Number of training epoch
        kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
        kernel_sizeSpa (int): Size of the spatial kernel
        filter_start (int): Number of output features of the first convolution layer
        sh_degree (int): Spherical harmonic degree of the fODF
        depth (int): Graph subsample depth
        n_side (int): Resolution of the Healpix map
        rf_name (str): Response function algorithm name
        wm (float): Use white matter
        gm (float): Use gray matter
        csf (float): Use CSF
        loss_fn_intensity (str): Name of the intensity loss
        loss_fn_non_negativity (str): Name of the nn loss
        loss_fn_sparsity (str): Name of the sparsity loss
        intensity_weight (float): Weight of the intensity loss
        nn_fodf_weight (float): Weight of the nn loss
        sparsity_weight (float): Weight of the sparsity loss
        save_path (str): Save path
        save_every (int): Frequency to save the model
        normalize (bool): Normalize the fODFs
        load_state (str): Load pre trained network
        patch_size (bool): Patch size neighborhood
    """
    
    # Load the shell and the graph samplings
    dataset = DMRIDataset(f'{data_path}/features.nii', f'{data_path}/mask.nii', f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', patch_size, concatenate=concatenate)
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
    dataloader_train = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    n_batch = len(dataloader_train)
    
    # Load the Polar filter used for the deconvolution
    polar_filter_equi, polar_filter_inva = load_response_function(f'{data_path}/response_functions/{rf_name}', wm=wm, gm=gm, csf=csf, max_degree=sh_degree, n_shell=len(shellSampling.shell_values), norm=dataset.normalization_value)

    # Create the deconvolution model
    model = Model(polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in)
    if load_state:
        print(load_state)
        model.load_state_dict(torch.load(load_state), strict=False)
    # Load model in GPU
    model = model.to(DEVICE)
    torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_0.pth'))

    # Loss
    intensity_criterion = Loss(loss_fn_intensity)
    non_negativity_criterion = Loss(loss_fn_non_negativity)
    sparsity_criterion = Loss(loss_fn_sparsity, sigma_sparsity)
    # Create dense interpolation used for the non-negativity and the sparsity losses
    denseGrid_interpolate = ComputeSignal(torch.Tensor(graphSampling.sampling.SH2S))
    denseGrid_interpolate = denseGrid_interpolate.to(DEVICE)

    # Optimizer/Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, threshold=0.01, factor=0.1, patience=3, verbose=True)
    save_loss = {}
    save_loss['train'] = {}
    writer = SummaryWriter(log_dir=os.path.join(data_path, 'result', 'run', save_path.split('/')[-1]))
    n_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if use_wandb:
        wandb.log({'learnable_params': n_params})
        wandb.watch(model, log="all")
    tb_j = 0
    # Training loop
    for epoch in range(n_epoch):
        # TRAIN
        model.train()

        # Initialize loss to save and plot.
        loss_intensity_ = 0
        loss_sparsity_ = 0
        loss_non_negativity_fodf_ = 0
        loss_pve_equi_ = 0
        loss_pve_inva_ = 0

        # Train on batch.
        for batch, data in enumerate(dataloader_train):
            start = time.time()
            to_wandb = {'epoch': epoch + 1, 'batch': tb_j}
            # Delete all previous gradients
            optimizer.zero_grad()
            to_print = ''

            # Load the data in the DEVICE
            input = data['input'].to(DEVICE)
            mask = data['mask'].to(DEVICE)
            output = data['out'].to(DEVICE)

            x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(input)
            if middle_voxel:
                x_reconstructed = x_reconstructed[:, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                x_deconvolved_equi_shc = x_deconvolved_equi_shc[:, :, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                x_deconvolved_inva_shc = x_deconvolved_inva_shc[:, :, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                output = output[:, :, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
                mask = mask[:, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1, patch_size//2:patch_size//2+1]
            ###############################################################################################
            ###############################################################################################
            # Loss
            ###############################################################################################
            ###############################################################################################
            # Intensity loss
            loss_intensity = intensity_criterion(x_reconstructed, output, mask[:, None].expand(-1, output.shape[1], -1, -1, -1))
            loss_intensity_ += loss_intensity.item()
            loss = intensity_weight * loss_intensity
            to_print += ', Intensity: {0:.10f}'.format(loss_intensity.item())
            to_wandb[f'Batch/train_intensity'] = loss_intensity.item()

            if not x_deconvolved_equi_shc  is None:
                x_deconvolved_equi = denseGrid_interpolate(x_deconvolved_equi_shc)
                ###############################################################################################
                # Sparsity loss
                equi_sparse = torch.zeros(x_deconvolved_equi.shape).to(DEVICE)
                loss_sparsity = sparsity_criterion(x_deconvolved_equi, equi_sparse, mask[:, None, None].expand(-1, equi_sparse.shape[1], equi_sparse.shape[2], -1, -1, -1))
                loss_sparsity_ += loss_sparsity.item()
                loss += sparsity_weight * loss_sparsity
                to_print += ', Equi Sparsity: {0:.10f}'.format(loss_sparsity.item())
                to_wandb[f'Batch/train_sparsity'] = loss_sparsity.item()

                ###############################################################################################
                # Non negativity loss
                fodf_neg = torch.min(x_deconvolved_equi, torch.zeros_like(x_deconvolved_equi))
                fodf_neg_zeros = torch.zeros(fodf_neg.shape).to(DEVICE)
                loss_non_negativity_fodf = non_negativity_criterion(fodf_neg, fodf_neg_zeros, mask[:, None, None].expand(-1, fodf_neg_zeros.shape[1], fodf_neg_zeros.shape[2], -1, -1, -1))
                loss_non_negativity_fodf_ += loss_non_negativity_fodf.item()
                loss += nn_fodf_weight * loss_non_negativity_fodf
                to_print += ', Equi NN: {0:.10f}'.format(loss_non_negativity_fodf.item())
                to_wandb[f'Batch/train_equi_nn'] = loss_non_negativity_fodf.item()

                ###############################################################################################
                # Partial volume regularizer
                loss_pve_equi = 1/(torch.mean(x_deconvolved_equi_shc[:, :, 0][mask[:, None].expand(-1, x_deconvolved_equi_shc.shape[1], -1, -1, -1)==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_equi_ += loss_pve_equi.item()
                loss += pve_weight * loss_pve_equi
                to_print += ', Equi regularizer: {0:.10f}'.format(loss_pve_equi.item())
                to_wandb[f'Batch/train_pve_wm'] = loss_pve_equi.item()

            #if not x_deconvolved_inva_shc is None:
            #    ###############################################################################################
            #    # Partial volume regularizer
            #    loss_pve_inva = 1/torch.mean(x_deconvolved_inva_shc[:, :, 0][mask[:, None].expand(-1, x_deconvolved_inva_shc.shape[1], -1, -1, -1)==1])*np.sqrt(4*np.pi)
            #    loss_pve_inva_ += loss_pve_inva.item()
            #    loss += pve_weight * loss_pve_inva
            #    to_print += ', Inva regularizer: {0:.10f}'.format(loss_pve_inva.item())
            ###############################################################################################
                # Partial volume regularizer
            index = 0
            if gm:
                loss_pve_inva = 1/(torch.mean(x_deconvolved_inva_shc[:, index, 0][mask==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_inva_ += loss_pve_inva.item()
                loss += pve_weight * loss_pve_inva
                to_print += ', Inva regularizer GM: {0:.10f}'.format(loss_pve_inva.item())
                to_wandb[f'Batch/train_pve_gm'] = loss_pve_inva.item()
                index += 1
            if csf:
                loss_pve_inva = 1/(torch.mean(x_deconvolved_inva_shc[:, index, 0][mask==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_inva_ += loss_pve_inva.item()
                loss += pve_weight * loss_pve_inva
                to_print += ', Inva regularizer CSF: {0:.10f}'.format(loss_pve_inva.item())
                to_wandb[f'Batch/train_pve_csf'] = loss_pve_inva.item()
            ###############################################################################################
            # Tensorboard
            tb_j += 1
            writer.add_scalar('Batch/train_intensity', loss_intensity.item(), tb_j)
            writer.add_scalar('Batch/train_sparsity', loss_sparsity.item(), tb_j)
            writer.add_scalar('Batch/train_nn', loss_non_negativity_fodf.item(), tb_j)
            writer.add_scalar('Batch/train_pve_equi', loss_pve_equi.item(), tb_j)
            writer.add_scalar('Batch/train_pve_inva', loss_pve_inva.item(), tb_j)
            writer.add_scalar('Batch/train_total', loss.item(), tb_j)
            to_wandb[f'Batch/train_total'] = loss.item()

            ###############################################################################################
            # Loss backward
            loss = loss
            loss.backward()
            optimizer.step()

            ###############################################################################################
            # To print loss
            end = time.time()
            to_print += ', Elapsed time: {0} s'.format(end - start)
            to_print = 'Epoch [{0}/{1}], Iter [{2}/{3}]: Loss: {4:.10f}'.format(epoch + 1, n_epoch,
                                                                                batch + 1, n_batch,
                                                                                loss.item()) \
                       + to_print
            print(to_print, end="\r")

            if (batch + 1) % 500 == 0:
                torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_{0}.pth'.format(epoch + 1)))
            if use_wandb:
                wandb.log(to_wandb)
        ###############################################################################################
        # Save and print mean loss for the epoch
        print("")
        to_print = ''
        loss_ = 0
        # Mean results of the last epoch
        save_loss['train'][epoch] = {}

        save_loss['train'][epoch]['loss_intensity'] = loss_intensity_ / n_batch
        save_loss['train'][epoch]['weight_loss_intensity'] = intensity_weight
        loss_ += intensity_weight * loss_intensity_
        to_print += ', Intensity: {0:.10f}'.format(loss_intensity_ / n_batch)

        save_loss['train'][epoch]['loss_sparsity'] = loss_sparsity_ / n_batch
        save_loss['train'][epoch]['weight_loss_sparsity'] = sparsity_weight
        loss_ += sparsity_weight * loss_sparsity_
        to_print += ', Sparsity: {0:.10f}'.format(loss_sparsity_ / n_batch)

        save_loss['train'][epoch]['loss_non_negativity_fodf'] = loss_non_negativity_fodf_ / n_batch
        save_loss['train'][epoch]['weight_loss_non_negativity_fodf'] = nn_fodf_weight
        loss_ += nn_fodf_weight * loss_non_negativity_fodf_
        to_print += ', WM fODF NN: {0:.10f}'.format(loss_non_negativity_fodf_ / n_batch)

        save_loss['train'][epoch]['loss_pve_equi'] = loss_pve_equi_ / n_batch
        save_loss['train'][epoch]['weight_loss_pve_equi'] = pve_weight
        loss_ += pve_weight * loss_pve_equi_
        to_print += ', Equi regularizer: {0:.10f}'.format(loss_pve_equi_ / n_batch)

        save_loss['train'][epoch]['loss_pve_inva'] = loss_pve_inva_ / n_batch
        save_loss['train'][epoch]['weight_loss_pve_inva'] = pve_weight
        loss_ += pve_weight * loss_pve_inva_
        to_print += ', Inva regularizer: {0:.10f}'.format(loss_pve_inva_ / n_batch)

        save_loss['train'][epoch]['loss'] = loss_ / n_batch
        to_print = 'Epoch [{0}/{1}], Train Loss: {2:.10f}'.format(epoch + 1, n_epoch, loss_ / n_batch) + to_print
        print(to_print)

        writer.add_scalar('Epoch/train_intensity', loss_intensity_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_sparsity', loss_sparsity_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_nn', loss_non_negativity_fodf_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_pve_equi', loss_pve_equi_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_pve_inva', loss_pve_inva_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_total', loss_ / n_batch, epoch)

        to_wandb = {'epoch': epoch + 1, 'Epoch/learning_rate': scheduler.optimizer.param_groups[0]['lr'],
                    'Epoch/train_total': loss_ / n_batch,
                    'Epoch/train_intensity': loss_intensity_ / n_batch,
                    'Epoch/train_sparsity': loss_sparsity_ / n_batch,
                    'Epoch/train_nn': loss_non_negativity_fodf_ / n_batch,
                    'Epoch/train_pve_equi': loss_pve_equi_ / n_batch,
                    'Epoch/train_pve_inva': loss_pve_inva_ / n_batch}
        if use_wandb:
            wandb.log(to_wandb)

        ###############################################################################################
        # VALIDATION
        scheduler.step(loss_ / n_batch)
        if epoch == 0:
            min_loss = loss_
            epochs_no_improve = 0
            n_epochs_stop = 5
            early_stop = False
        elif loss_ < min_loss * 0.999:
            epochs_no_improve = 0
            min_loss = loss_
        else:
            epochs_no_improve += 1
        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True

        ###############################################################################################
        # Save the loss and model
        with open(os.path.join(save_path, 'history', 'loss.pkl'), 'wb') as f:
            pickle.dump(save_loss, f)
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_{0}.pth'.format(epoch + 1)))
        if early_stop:
            print("Stopped")
            break


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
        '--lr',
        default=1e-2,
        help='Learning rate (default: 1e-2)',
        type=float
    )
    parser.add_argument(
        '--epoch',
        default=100,
        help='Epoch (default: 100)',
        type=int
    )
    parser.add_argument(
        '--filter_start',
        help='Number of filters for the first convolution (default: 8)',
        default=8,
        type=int
    )
    parser.add_argument(
        '--sh_degree',
        help='Max spherical harmonic order (default: 20)',
        default=20,
        type=int
    )
    parser.add_argument(
        '--kernel_sizeSph',
        help='Spherical kernel size (default: 5)',
        default=5,
        type=int
    )
    parser.add_argument(
        '--kernel_sizeSpa',
        help='Spatial kernel size (default: 3)',
        default=3,
        type=int
    )
    parser.add_argument(
        '--depth',
        help='Graph subsample depth (default: 5)',
        default=5,
        type=int
    )
    parser.add_argument(
        '--n_side',
        help='Healpix resolution (default: 16)',
        default=16,
        type=int
    )
    parser.add_argument(
        '--save_every',
        help='Saving periodicity (default: 2)',
        default=2,
        type=int
    )
    parser.add_argument(
        '--loss_intensity',
        choices=('L1', 'L2'),
        default='L2',
        help='Objective function (default: L2)',
        type=str
    )
    parser.add_argument(
        '--intensity_weight',
        default=1.,
        help='Intensity weight (default: 1.)',
        type=float
    )
    parser.add_argument(
        '--loss_sparsity',
        choices=('L1', 'L2', 'cauchy', 'welsch', 'geman'),
        default='cauchy',
        help='Objective function (default: cauchy)',
        type=str
    )
    parser.add_argument(
        '--sigma_sparsity',
        default=1e-5,
        help='Sigma for sparsity (default: 1e-5)',
        type=float
    )
    parser.add_argument(
        '--sparsity_weight',
        default=1.,
        help='Sparsity weight (default: 1.)',
        type=float
    )
    parser.add_argument(
        '--loss_non_negativity',
        choices=('L1', 'L2'),
        default='L2',
        help='Objective function (default: L2)',
        type=str
    )
    parser.add_argument(
        '--nn_fodf_weight',
        default=1.,
        help='Non negativity fODF weight (default: 1.)',
        type=float
    )
    parser.add_argument(
        '--pve_weight',
        default=1e-5,
        help='PVE regularizer weight (default: 1e-5)',
        type=float
    )
    parser.add_argument(
        '--load_state',
        help='Load a saved model (default: None)',
        type=str
    )
    parser.add_argument(
        '--rf_name',
        required=True,
        help='Response function folder name (default: None)',
        type=str
    )
    parser.add_argument(
        '--wm',
        required=True,
        action='store_true',
        help='Estimate white matter fODF (default: False)',
    )
    parser.add_argument(
        '--gm',
        action='store_true',
        help='Estimate grey matter fODF (default: False)',
    )
    parser.add_argument(
        '--csf',
        action='store_true',
        help='Estimate CSF fODF (default: False)',
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Norm the partial volume sum to be 1 (default: False)',
    )
    parser.add_argument(
        '--patch_size',
        default=1,
        help='Patch size (default: 1)',
        type=int
    )
    parser.add_argument(
        '--graph_sampling',
        default='healpix',
        choices=('healpix', 'bvec'),
        help='Sampling used for the graph convolution, healpix or bvec (default: healpix)',
        type=str
    )
    parser.add_argument(
        '--conv_name',
        default='mixed',
        choices=('mixed', 'spherical'),
        help='Convolution name (default: mixed)',
        type=str
    )
    parser.add_argument(
        '--anisoSpa',
        action='store_true',
        help='Use anisotropic spatial filter (default: False)',
    )
    parser.add_argument(
        '--concatenate',
        action='store_true',
        help='Concatenate spherical features (default: False)',
    )
    parser.add_argument(
        '--project',
        default='default',
        help='Project name',
        type=str
    )
    parser.add_argument(
        '--expname',
        default='default',
        help='Exp name',
        type=str
    )
    parser.add_argument(
        '--middle_voxel',
        action='store_true',
        help='Concatenate spherical features (default: False)',
    )
    args = parser.parse_args()
    data_path = args.data_path
    expname = args.expname
    middle_voxel = args.middle_voxel
    
    # Train properties
    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.epoch

    # Model architecture properties
    filter_start = args.filter_start
    sh_degree = args.sh_degree
    kernel_sizeSph = args.kernel_sizeSph
    kernel_sizeSpa = args.kernel_sizeSpa
    depth = args.depth
    n_side = args.n_side
    normalize = args.normalize
    patch_size = args.patch_size
    graph_sampling = args.graph_sampling
    conv_name = args.conv_name
    isoSpa = not args.anisoSpa
    concatenate = args.concatenate

    # Saving parameters
    save_path = os.path.join(data_path, 'result')
    save_every = args.save_every

    # Intensity loss
    loss_fn_intensity = args.loss_intensity
    intensity_weight = args.intensity_weight
    # Sparsity loss
    loss_fn_sparsity = args.loss_sparsity
    sigma_sparsity = args.sigma_sparsity
    sparsity_weight = args.sparsity_weight
    # Non-negativity loss
    loss_fn_non_negativity = args.loss_non_negativity
    nn_fodf_weight = args.nn_fodf_weight
    # PVE loss
    pve_weight = args.pve_weight

    # Load pre-trained model and response functions
    load_state = args.load_state
    rf_name = args.rf_name
    wm = args.wm
    gm = args.gm
    csf = args.csf

    if use_wandb:
        wandb.init(project=args.project, entity='axelelaldi')
        config = wandb.config

        config.data_path = data_path
        config.expname = expname
        config.middle_voxel = middle_voxel
        
        # Train properties
        config.batch_size = batch_size
        config.lr = lr
        config.n_epoch = n_epoch
        
        # Model architecture properties
        config.filter_start = filter_start
        config.sh_degree = sh_degree
        config.kernel_sizeSph = kernel_sizeSph
        config.kernel_sizeSpa = kernel_sizeSpa
        config.depth = depth
        config.n_side = n_side
        config.normalize = normalize
        config.patch_size = patch_size
        config.graph_sampling = graph_sampling
        config.conv_name = conv_name
        config.isoSpa = isoSpa
        config.concatenate = concatenate

        # Saving parameters
        config.save_every = save_every

        # Intensity loss
        config.loss_fn_intensity = loss_fn_intensity
        config.intensity_weight = intensity_weight
        # Sparsity loss
        config.loss_fn_sparsity = loss_fn_sparsity
        config.sigma_sparsity = sigma_sparsity
        config.sparsity_weight = sparsity_weight
        # Non-negativity loss
        config.loss_fn_non_negativity = loss_fn_non_negativity
        config.nn_fodf_weight = nn_fodf_weight
        # PVE loss
        config.pve_weight = pve_weight

        # Load pre-trained model and response functions
        config.load_state = load_state
        config.rf_name = rf_name
        config.wm = wm
        config.gm = gm
        config.csf = csf


    # Save directory
    if not os.path.exists(save_path):
        print('Create new directory: {0}'.format(save_path))
        os.makedirs(save_path)
    save_path = os.path.join(save_path, time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime()))
    save_path += f'_{conv_name}_{isoSpa}_{kernel_sizeSph}_{kernel_sizeSpa}_{n_side}_{patch_size}_{sparsity_weight}_{pve_weight}_{filter_start}_{depth}_{sh_degree}_{expname}_{middle_voxel}_{wm}_{gm}_{csf}'
    print('Save path: {0}'.format(save_path))

    # History directory
    history_path = os.path.join(save_path, 'history')
    if not os.path.exists(history_path):
        print('Create new directory: {0}'.format(history_path))
        os.makedirs(history_path)

    if use_wandb:
        config.save_path = save_path

    # Save parameters
    with open(os.path.join(save_path, 'args.txt'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    main(data_path, batch_size, lr, n_epoch, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         loss_fn_intensity, loss_fn_non_negativity, loss_fn_sparsity, sigma_sparsity,
         intensity_weight, nn_fodf_weight, sparsity_weight, pve_weight,
         save_path, save_every, normalize, load_state, patch_size, graph_sampling, conv_name, isoSpa, concatenate, middle_voxel)

