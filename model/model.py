import torch
from .deconvolution import Deconvolution
from .reconstruction import Reconstruction


class Model(torch.nn.Module):
    def __init__(self, polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in=1):
        super(Model, self).__init__()
        
        n_equi = polar_filter_equi.shape[0]
        n_inva = polar_filter_inva.shape[0]
        self.deconvolution = Deconvolution(shellSampling, graphSampling, filter_start, kernel_sizeSph, kernel_sizeSpa, n_equi, n_inva, normalize, conv_name, isoSpa, feature_in)
        self.reconstruction = Reconstruction(polar_filter_equi, polar_filter_inva, shellSampling)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V' x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x V' x X x Y x Z]
            :obj:`torch.Tensor`: output [B x out_channels_equi x C x X x Y x Z]
            :obj:`torch.Tensor`: output [B x out_channels_inva x 1 x X x Y x Z]
        """
        # Deconvolve the signal and get the spherical harmonic coefficients
        x_deconvolved_equi_shc, x_deconvolved_inva_shc = self.deconvolution(x) # B x out_channels_equi x C x X x Y x Z, B x out_channels_inva x 1 x X x Y x Z

        # Reconstruct the signal
        x_reconstructed = self.reconstruction(x_deconvolved_equi_shc, x_deconvolved_inva_shc) # B x V' x X x Y x Z

        return x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc
