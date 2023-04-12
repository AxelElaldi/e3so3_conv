from utils.dataset import mnist_dataset
from model.unet import GraphCNNUnet
from utils.sampling import HealpixSampling, _sh_matrix

import os
import time
import pickle
import json
import argparse

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


use_wandb = False
if use_wandb:
    import wandb

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, sfx_train, batch_size, bandwidth, depth,
         kernel_sizeSph, kernel_sizeSpa, lr, n_epoch, pooling_mode,
         save_every, validation, conv_name, save_path, isoSpa, start_filter, crop='local', cropsize=14, background=False):
    
    # MODEL
    if background:
        out_channels = 10
    else:
        out_channels = 11
    block_depth = 2
    in_depth = 1
    keepSphericalDim = False
    patch_size = 16
    sh_degree = 6
    pooling_name = conv_name
    hp = HealpixSampling(bandwidth, depth, patch_size, sh_degree, pooling_mode, pooling_name)
    if conv_name=='spatial_sh':
        S2SH, _ = _sh_matrix(sh_degree, hp.sampling.vectors, with_order=1, symmetric=False)
        S2SH = torch.Tensor(S2SH).to(DEVICE)
        in_channels = S2SH.shape[1]
    elif conv_name=='spatial_vec':
        S2SH, _ = _sh_matrix(sh_degree, hp.sampling.vectors, with_order=1, symmetric=False)
        S2SH = torch.Tensor(S2SH).to(DEVICE)
        in_channels = S2SH.shape[0]
    else:
        in_channels = 1
    model = GraphCNNUnet(in_channels, out_channels, start_filter, block_depth, in_depth, kernel_sizeSph, kernel_sizeSpa, hp.pooling, hp.laps, conv_name, isoSpa, keepSphericalDim, hp.vec)
    model = model.to(DEVICE)
    
    # RECORD NUMBER OF TRAINABLE PARAMETERS
    n_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print("#params ", n_params)
    if use_wandb:
        wandb.log({'learnable_params': n_params})
    tb_j = 0
    writer = SummaryWriter(log_dir=os.path.join(data_path, 'result', 'run', save_path.split('/')[-1]))
    writer.add_scalar('Epoch/learnable_params', n_params, tb_j)
    
    # DATASET 
    dataset_train = mnist_dataset(data_path, sfx_train, bandwidth, 'train', crop=crop, cropsize=cropsize, background=background)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    n_batch = len(dataloader_train)

    dataloader_val_list = []
    n_batch_val_list = []
    sfx_val_list = ['norot', 'voxel', 'grid', 'gridvoxel', 'gridvoxelsame']
    for sfx_val in sfx_val_list:
        dataset_val = mnist_dataset(data_path, sfx_val, bandwidth, 'val', crop=crop, cropsize=cropsize, background=background)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
        n_batch_val = len(dataloader_val)
        dataloader_val_list.append(dataloader_val)
        n_batch_val_list.append(n_batch_val)

    # CLASS WEIGHTS FOR BALANCED LOSS
    fixed_weight = True
    if fixed_weight:
        if background:
            w = 1/torch.Tensor([0.5] + [0.5/(out_channels-1) for _ in range(out_channels-1)])
        else:
            w = 1/(torch.Tensor([0.5/(out_channels-1) for _ in range(out_channels-1)] + [0.5]))
        w = w / torch.sum(w)
        w = w.to(DEVICE)

    # Optimizer/Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[25, 35, 45], gamma=0.5)
    save_loss = {}
    save_loss['train'] = {}
    save_loss['val'] = {}
    
    lossCE = torch.nn.CrossEntropyLoss()
    if use_wandb:
        wandb.watch(model, lossCE, log="all")
    sft = torch.nn.Softmax(dim=1)
    # Training loop
    for epoch in range(n_epoch):
        # TRAIN
        model.train()

        # Initialize loss to save and plot.
        ce_loss_ = 0
        dice_loss_ = 0
        accuracy_ = torch.zeros(out_channels)

        # Train on batch.
        for batch, data in enumerate(dataloader_train):
            start = time.time()
            to_wandb = {'epoch': epoch + 1, 'batch': tb_j}
            # Delete all previous gradients
            optimizer.zero_grad()
            to_print = ''

            # Load the data in the DEVICE
            input = data['image']['data']
            input = input[:, None].type(torch.float32).to(DEVICE)
            seg_gt = data['label']['data']
            seg_gt = seg_gt[:, 0].type(torch.LongTensor).to(DEVICE)
            if conv_name in ['spatial_sh', 'spatial_vec']:
                if conv_name=='spatial_sh':
                    input = input.permute(0, 1, 3, 4, 5, 2).matmul(S2SH).permute(0, 1, 5, 2, 3, 4)
                B, C, V, X, Y, Z = input.shape
                input = input.view(B, C*V, 1, X, Y, Z)
            seg_pred = model(input).squeeze(2)
            ###############################################################################################
            ###############################################################################################
            # Loss
            ###############################################################################################
            ###############################################################################################
            # https://arxiv.org/pdf/1707.03237.pdf
            # compute the dice score
            dims = (0, 2, 3, 4)
            seg_soft = sft(seg_pred)
            target_one_hot = torch.nn.functional.one_hot(seg_gt, num_classes=seg_pred.shape[1]).permute(0, 4, 1, 2, 3)
            if not fixed_weight:
                w = 1/(torch.sum(target_one_hot, dims)**2 + 1e-5)

            inter = w * torch.sum(seg_soft * target_one_hot, dims)
            union = w * torch.sum(seg_soft + target_one_hot, dims)
            dice_loss = 1 - (2 * torch.sum(inter) + 1e-5) / (torch.sum(union) + 1e-5)
            dice_loss_ += dice_loss.item()
            to_print += f' Dice loss: {dice_loss.item():.3f}'

            # Cross Entropy loss
            if not fixed_weight:
                w = 1/(torch.mean(target_one_hot.type(torch.float32).detach(), dims) + 1e-5) - 1
            ce_loss = torch.nn.functional.cross_entropy(seg_pred, seg_gt, w)
            ce_loss_ += ce_loss.item()
            to_print += f' CE loss: {ce_loss.item():.3f}'

            loss = 0.5 * dice_loss + (1 - 0.5) * ce_loss

            accuracy = torch.zeros(out_channels)
            accuracy_all = ((torch.argmax(seg_soft, dim=1))==seg_gt).type(torch.float)
            for c in range(out_channels):
                accuracy[c] = torch.mean(accuracy_all[seg_gt==c])
                to_print += f', Accuracy {c}: {100*accuracy[c]:.1f}'
                to_wandb[f'Batch/train_sensitivity_{c}'] = accuracy[c]
            accuracy_ += accuracy

            ###############################################################################################
            # Tensorboard
            tb_j += 1
            writer.add_scalar('Batch/train_total', loss.item(), tb_j)
            writer.add_scalar('Batch/train_dice', dice_loss.item(), tb_j)
            writer.add_scalar('Batch/train_ce', ce_loss.item(), tb_j)
            for c in range(out_channels):
                writer.add_scalar(f'Batch/train_accuracy_{c}', accuracy[c], tb_j)

            ###############################################################################################
            # Loss backward
            loss.backward()
            optimizer.step()

            ###############################################################################################
            # To print loss
            end = time.time()
            to_wandb['Batch/train_speed'] = end - start
            to_wandb['Batch/train_total'] = loss.item()
            to_wandb['Batch/train_dice'] = dice_loss.item()
            to_wandb['Batch/train_ce'] = ce_loss.item()
            writer.add_scalar('Batch/train_speed', end - start, tb_j)
            to_print += f', Elapsed time: {end - start:.3f} s'
            to_print = f'Epoch [{epoch + 1}/{n_epoch}], Iter [{batch + 1}/{n_batch}]:' \
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
        to_wandb = {'epoch': epoch + 1, 'Epoch/learning_rate': scheduler.optimizer.param_groups[0]['lr'],
                    'Epoch/train_total': (0.5 * dice_loss_ + (1-0.5) * ce_loss_) / n_batch,
                    'Epoch/train_dice': dice_loss_ / n_batch,
                    'Epoch/train_ce': ce_loss_ / n_batch}

        # Mean results of the last epoch
        save_loss['train'][epoch] = {}
        save_loss['train'][epoch]['accuracy'] = accuracy_ / n_batch
        for c in range(out_channels):
            to_print += f', Train Accuracy {c}: {100*accuracy_[c] / n_batch:.1f}'
        save_loss['train'][epoch]['dice'] = dice_loss_ / n_batch
        to_print = f', Train dice Loss: {dice_loss_ / n_batch:.3f}' + to_print
        save_loss['train'][epoch]['ce'] = ce_loss_ / n_batch
        to_print = f', Train ce Loss: {ce_loss_ / n_batch:.3f}' + to_print
        save_loss['train'][epoch]['loss'] = (0.5 * dice_loss_ + (1-0.5) * ce_loss_) / n_batch
        to_print = f'Epoch [{epoch + 1}/{n_epoch}], Train Loss: {(0.5 * dice_loss_ + (1-0.5) * ce_loss_) / n_batch:.3f}' + to_print
        print(to_print)

        writer.add_scalar('Epoch/train_dice', dice_loss_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_ce', ce_loss_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_total', (0.5 * dice_loss_ + (1-0.5) * ce_loss_) / n_batch, epoch)
        for c in range(out_channels):
            writer.add_scalar(f'Epoch/train_accuracy_{c}', accuracy_[c] / n_batch, epoch)
            to_wandb[f'Epoch/train_accuracy_{c}'] = accuracy_[c] / n_batch
        writer.add_scalar('Epoch/learning_rate', scheduler.optimizer.param_groups[0]['lr'], epoch)
        if use_wandb:
            wandb.log(to_wandb)
        print("")
        ###############################################################################################
        # VALIDATION
        with torch.no_grad():
            if validation:
                to_wandb = {'epoch': epoch + 1}
                for sfx_val, n_batch_val, dataloader_val in zip(sfx_val_list, n_batch_val_list, dataloader_val_list):
                    elapsed_ = 0
                    sensitivity_ = torch.zeros(len(dataset_val), out_channels)
                    dice_ = torch.zeros(len(dataset_val), out_channels)
                    ce_ = torch.zeros(len(dataset_val), out_channels)
                    ce_loss_ = torch.zeros(len(dataset_val))
                    dice_loss_ = torch.zeros(len(dataset_val))
                    loss_ = torch.zeros(len(dataset_val))
                    model.eval()
                    s = 0
                    # Train on batch.
                    for batch, data in enumerate(dataloader_val):
                        start = time.time()
                        # Load the data in the DEVICE
                        #input = data['input'].to(DEVICE)
                        #seg_gt = data['output'].to(DEVICE)
                        input = data['image']['data']
                        input = input[:, None].type(torch.float32).to(DEVICE)
                        seg_gt = data['label']['data']
                        seg_gt = seg_gt[:, 0].type(torch.LongTensor).to(DEVICE)
                        if conv_name in ['spatial_sh', 'spatial_vec']:
                            if conv_name=='spatial_sh':
                                input = input.permute(0, 1, 3, 4, 5, 2).matmul(S2SH).permute(0, 1, 5, 2, 3, 4)
                            B, C, V, X, Y, Z = input.shape
                            input = input.view(B, C*V, 1, X, Y, Z)
                        seg_pred = model(input).squeeze(2)
                        ###############################################################################################
                        ###############################################################################################
                        # Loss
                        ###############################################################################################
                        ###############################################################################################
                        # https://arxiv.org/pdf/1707.03237.pdf
                        # compute the dice score
                        dims = (2, 3, 4)
                        seg_soft = sft(seg_pred)
                        target_one_hot = torch.nn.functional.one_hot(seg_gt, num_classes=seg_soft.shape[1]).permute(0, 4, 1, 2, 3)
                        if not fixed_weight:
                            w = 1/(torch.sum(target_one_hot, dims)**2 + 1e-5)

                        inter = w * torch.sum(seg_soft * target_one_hot, dims)
                        union = w * torch.sum(seg_soft + target_one_hot, dims)

                        dice_loss = 1 - (2 * torch.sum(inter, axis=1) + 1e-5) / (torch.sum(union, axis=1) + 1e-5)

                        # Cross Entropy loss
                        dims = (0, 2, 3, 4)
                        if not fixed_weight:
                            w = 1/(torch.mean(target_one_hot.type(torch.float32).detach(), dims) + 1e-5) - 1
                        ce_loss = torch.mean(torch.nn.functional.cross_entropy(seg_pred, seg_gt, w, reduction='none'), (1, 2, 3))

                        ce_loss_[s:s+seg_soft.shape[0]] = ce_loss
                        dice_loss_[s:s+seg_soft.shape[0]] = dice_loss
                        loss_[s:s+seg_soft.shape[0]] = 0.5 * dice_loss + (1 - 0.5) * ce_loss

                        sensitivity_all = (torch.argmax(seg_soft, dim=1)==seg_gt).type(torch.float)
                        for c in range(out_channels):
                            for s2 in range(seg_soft.shape[0]):
                                sensitivity_[s+s2, c] = torch.mean(sensitivity_all[s2][seg_gt[s2]==c]) if sensitivity_all[s2][seg_gt[s2]==c].shape[0] != 0 else 1
                            dice_[s:s+seg_soft.shape[0], c] = 1 - (2 * inter[:, c] + 1e-5) / (union[:, c] + 1e-5)
                        s += seg_soft.shape[0]

                        ###############################################################################################
                        # To print loss
                        end = time.time()
                        elapsed_ += end - start

                    to_print = ''
                    for c in range(out_channels):
                        to_print += f', Val Accuracy {c}: {100*torch.mean(sensitivity_[:, c]):.1f}'
                        #to_print += f', Val Dice {c}: {torch.mean(dice_[:, c]):.1f}'
                        to_wandb[f'Epoch/val_{sfx_val}_sensitivity_{c}'] = torch.mean(sensitivity_[:, c])
                        to_wandb[f'Epoch/val_{sfx_val}_dice_{c}'] = torch.mean(dice_[:, c])
                        to_wandb[f'Epoch/val_{sfx_val}_sensitivity_std_{c}'] = torch.std(sensitivity_[:, c])
                        to_wandb[f'Epoch/val_{sfx_val}_dice_std_{c}'] = torch.std(dice_[:, c])
                    to_print = f', Val dice loss: {torch.mean(dice_loss_):.3f}' + to_print
                    to_print = f', Val ce loss: {torch.mean(ce_loss_):.3f}' + to_print
                    to_print = f'Epoch [{epoch + 1}/{n_epoch}], {sfx_val}, Val Loss: {torch.mean(loss_):.3f}' + to_print
                    to_wandb[f'Epoch/val_{sfx_val}_sensitivity'] = torch.mean(sensitivity_)
                    to_wandb[f'Epoch/val_{sfx_val}_sensitivity_std'] = torch.std(sensitivity_)
                    to_wandb[f'Epoch/val_{sfx_val}_dice'] = torch.mean(dice_)
                    to_wandb[f'Epoch/val_{sfx_val}_dice_std'] = torch.std(dice_)
                    to_wandb[f'Epoch/val_{sfx_val}_loss'] = torch.mean(loss_)
                    to_wandb[f'Epoch/val_{sfx_val}_loss_std'] = torch.std(loss_)
                    print(to_print)
                print('')
                if use_wandb:
                    wandb.log(to_wandb)
        #scheduler.step(loss_ / n_batch)
        scheduler.step()
        if epoch == 0:
            min_loss = torch.mean(loss_)
            epochs_no_improve = 0
            n_epochs_stop = 15
            early_stop = False
        elif torch.mean(loss_) < min_loss * 0.999:
            epochs_no_improve = 0
            min_loss = torch.mean(loss_)
        else:
            epochs_no_improve += 1
        if epoch > 50 and epochs_no_improve == n_epochs_stop:
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
        '--sfx_train',
        default='no',
        help='Split name for training (default: no)',
        type=str
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        help='Batch size (default: 1)',
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
        default=50,
        help='Epoch (default: 50)',
        type=int
    )
    parser.add_argument(
        '--conv_name',
        default='mixed',
        help='Graph convolution name mixed or spherical or muller (default: mixed)',
        type=str
    )
    parser.add_argument(
        '--kernel_sizeSph',
        help='Spherical kernel size (default: 3)',
        default=3,
        type=int
    )
    parser.add_argument(
        '--kernel_sizeSpa',
        help='Spatial kernel size (default: 3)',
        default=3,
        type=int
    )
    parser.add_argument(
        '--anisoSpa',
        action='store_true',
        help='Use anisotropic spatial filter (default: False)',
    )
    parser.add_argument(
        '--depth',
        help='Graph subsample depth (default: 3)',
        default=3,
        type=int
    )
    parser.add_argument(
        '--start_filter',
        help='# output features first layer (default: 8)',
        default=8,
        type=int
    )
    parser.add_argument(
        '--bandwidth',
        help='Healpix resolution (default: 4)',
        default=4,
        type=int
    )
    parser.add_argument(
        '--save_every',
        help='Saving periodicity (default: 1)',
        default=1,
        type=int
    )
    parser.add_argument(
        '--no_validation',
        action='store_false',
        help='Track test loss and accuracy (default: True)',
    )
    parser.add_argument(
        '--cropsize',
        help='cropsize (default: 14)',
        default=14,
        type=int
    )
    parser.add_argument(
        '--crop',
        default='local',
        help='crop (default: local)',
        type=str
    )
    parser.add_argument(
        '--background',
        action='store_true',
        help='Dataset with fixed background of 0 (default: False)',
    )

    args = parser.parse_args()
    data_path = args.data_path
    sfx_train = args.sfx_train
    crop = args.crop
    cropsize = args.cropsize
    background = args.background
    
    # Train properties
    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.epoch
    
    # Model architecture properties
    bandwidth = args.bandwidth
    depth = args.depth
    kernel_sizeSph = args.kernel_sizeSph
    kernel_sizeSpa = args.kernel_sizeSpa
    pooling_mode = 'average' # args.pooling_mode
    conv_name = args.conv_name
    isoSpa = not args.anisoSpa
    start_filter = args.start_filter

    validation = args.no_validation
    if use_wandb:
        wandb.init(project='yourprojectname', entity='yourentity')
        config = wandb.config
        config.sfx_train = sfx_train
        config.batch_size = batch_size
        config.lr = lr
        config.bandwidth = bandwidth
        config.kernel_sizeSph = kernel_sizeSph
        config.kernel_sizeSpa = kernel_sizeSpa
        config.conv_name = conv_name
        config.isoSpa = isoSpa
        config.start_filter = start_filter
        config.crop = crop
        config.cropsize = cropsize
        config.background = background
    
    # Saving parameters
    save_every = args.save_every

    save_path = f'{data_path}/result/'
    # Save directory
    if not os.path.exists(save_path):
        print('Create new directory: {0}'.format(save_path))
        os.makedirs(save_path)
    save_path = os.path.join(save_path, time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime()))
    save_path += f'_{sfx_train}_{batch_size}_{lr}_{bandwidth}_{kernel_sizeSph}_{kernel_sizeSpa}_{conv_name}_{isoSpa}_{start_filter}_{crop}_{cropsize}_{background}'
    print('Save path: {0}'.format(save_path))
    if use_wandb:
        config.save_path = save_path

    # History directory
    history_path = os.path.join(save_path, 'history')
    if not os.path.exists(history_path):
        print('Create new directory: {0}'.format(history_path))
        os.makedirs(history_path)

    # Save parameters
    with open(os.path.join(save_path, 'args.txt'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    main(data_path, sfx_train, batch_size, bandwidth, depth,
         kernel_sizeSph, kernel_sizeSpa, lr, n_epoch, pooling_mode,
         save_every, validation, conv_name, save_path, isoSpa, start_filter, crop, cropsize, background)
