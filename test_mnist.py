import argparse
import os
from utils.dataset import mnist_dataset
from utils.pooling import HealpixPooling
from model.unet import GraphCNNUnet
from utils.sampling import HealpixSampling, _sh_matrix

import json
import argparse
import torch
from torch.utils.data.dataloader import DataLoader
import os
import time
import nibabel as nib
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, model_name, batch_size, bandwidth, depth,
         kernel_sizeSph, kernel_sizeSpa, pooling_mode,
         conv_name, isoSpa, start_filter, crop, cropsize, background):

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
    model.load_state_dict(torch.load(f'{data_path}/result/{model_name}/history/epoch_{epoch}.pth'), strict=False)
    model = model.to(DEVICE)
    model.eval()

    # DATASET 
    dataloader_test_list = []
    n_batch_test_list = []
    sfx_test_list = ['norot', 'voxel', 'grid', 'gridvoxel', 'gridvoxelsame']
    for sfx_test in sfx_test_list:
        dataset_test = mnist_dataset(data_path, sfx_test, bandwidth, 'test', crop=crop, cropsize=cropsize, background=background) 
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
        n_batch_test = len(dataloader_test)
        dataloader_test_list.append(dataloader_test)
        n_batch_test_list.append(n_batch_test)

    # CLASS WEIGHTS FOR BALANCED LOSS
    fixed_weight = True
    if fixed_weight:
        if background:
            w = 1/torch.Tensor([0.5] + [0.5/(out_channels-1) for _ in range(out_channels-1)])
        else:
            w = 1/(torch.Tensor([0.5/(out_channels-1) for _ in range(out_channels-1)] + [0.5]))
        w = w / torch.sum(w)
        w = w.to(DEVICE)

    sft = torch.nn.Softmax(dim=1)
    # Test on batch.
    with torch.no_grad():
        for sfx_test, n_batch_test, dataloader_test in zip(sfx_test_list, n_batch_test_list, dataloader_test_list):
            elapsed_ = 0
            sensitivity_ = torch.zeros(len(dataset_test), out_channels)
            dice_ = torch.zeros(len(dataset_test), out_channels)
            ce_loss_ = torch.zeros(len(dataset_test))
            dice_loss_ = torch.zeros(len(dataset_test))
            loss_ = torch.zeros(len(dataset_test))
            model.eval()
            s = 0
            # Train on batch.
            for batch, data in enumerate(dataloader_test):
                start = time.time()
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

                prediction_list = seg_soft.detach().cpu().numpy().astype(np.float32)
                for sub_id in range(prediction_list.shape[0]):
                    img = nib.Nifti1Image(prediction_list[sub_id], np.eye(4))
                    os.makedirs(f"{data_path}/result/{model_name}/test/epoch_{epoch}/{sfx_test}/test/{data['name'][sub_id]}/", exist_ok=True)
                    nib.save(img, f"{data_path}/result/{model_name}/test/epoch_{epoch}/{sfx_test}/test/{data['name'][sub_id]}/prediction.nii")
    

                target_one_hot = torch.nn.functional.one_hot(seg_gt, num_classes=seg_soft.shape[1]).permute(0, 4, 1, 2, 3)
                if not fixed_weight:
                    w = 1/(torch.sum(target_one_hot, dims)**2 + 1e-5)

                inter = torch.sum(seg_soft * target_one_hot, dims)
                union = torch.sum(seg_soft + target_one_hot, dims)

                dice_loss = 1 - (2 * torch.sum(w * inter, axis=1) + 1e-5) / (torch.sum(w * union, axis=1) + 1e-5)

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
            to_print = f', Val dice loss: {torch.mean(dice_loss_):.3f}' + to_print
            to_print = f', Val ce loss: {torch.mean(ce_loss_):.3f}' + to_print
            to_print = f'{sfx_test}, Val Loss: {torch.mean(loss_):.3f}' + to_print
            print(to_print)
            

            np.savetxt(f"{data_path}/result/{model_name}/test/epoch_{epoch}/{sfx_test}/dice_test.txt", dice_)
            np.savetxt(f"{data_path}/result/{model_name}/test/epoch_{epoch}/{sfx_test}/sensitivity_test.txt", sensitivity_)
            np.savetxt(f"{data_path}/result/{model_name}/test/epoch_{epoch}/{sfx_test}/loss_test.txt", loss_)

            


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
        default=16,
        help='Batch size (default: 16)',
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
    args = parser.parse_args()
    # Test properties
    batch_size = args.batch_size
    
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
    bandwidth = int(args_json['bandwidth']) 
    depth = int(args_json['depth']) 
    kernel_sizeSph = int(args_json['kernel_sizeSph']) 
    kernel_sizeSpa = int(args_json['kernel_sizeSpa']) 
    pooling_mode = 'average' # args.pooling_mode
    conv_name = str(args_json['conv_name']) 
    isoSpa = not bool(args_json['anisoSpa']) 
    start_filter = int(args_json['start_filter']) 
    crop = str(args_json['crop'])
    cropsize = int(args_json['cropsize'])
    background = bool(args_json['background'])


    print(f'bandwidth: {bandwidth}')
    print(f'depth: {depth}')
    print(f'Kernel size spherical: {kernel_sizeSph}')
    print(f'Kernel size spatial: {kernel_sizeSpa}')
    print(f'Unet depth: {depth}')
    print(f'conv_name: {conv_name}')
    print(f'isoSpa: {isoSpa}')
    print(f'start_filter: {start_filter}')
    print(f'crop: {crop}')
    print(f'cropsize: {cropsize}')
    print(f'background: {background}')

    # Test directory
    test_path = f'{data_path}/result/{model_name}/test/epoch_{epoch}'
    if not os.path.exists(test_path):
        print('Create new directory: {0}'.format(test_path))
        os.makedirs(test_path)

    main(data_path, model_name, batch_size, bandwidth, depth,
         kernel_sizeSph, kernel_sizeSpa, pooling_mode,
         conv_name, isoSpa, start_filter, crop, cropsize, background)
    