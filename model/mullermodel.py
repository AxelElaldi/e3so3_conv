from equideepdmri.network.VoxelWiseSegmentationNetwork import VoxelWiseSegmentationNetwork
import torch.nn as nn
from equideepdmri.utils.q_space import Q_SamplingSchema
from equideepdmri.layers.layer_builders import build_pq_layer, build_p_layer, build_q_reduction_layer
import torch
import math


class Segmentation(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, kernel_sizeSpa, vec, pool, start_filter=32, isoSpa=True, input_features=1, n_class=11):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(Segmentation, self).__init__()
        self.pooling = pool.pooling
        if isoSpa:
            out_filter1 = [start_filter]
        else:
            out_filter1 = [start_filter, math.ceil(start_filter/4)]
            #out_filter1 = [16, 4]
        self.conv1 = build_pq_layer([input_features], out_filter1, #[1], [16, 4],
                p_kernel_size=kernel_sizeSpa,
                kernel="pq_TP",
                q_sampling_schema_in=vec[0],
                q_sampling_schema_out=vec[1],
                use_batch_norm=True,
                transposed=False,
                non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                use_non_linearity=True,
                p_radial_basis_type="cosine",
                p_radial_basis_params={"num_layers": 3, "num_units": 50},
                sub_kernel_selection_rule={"l_diff_to_out_max": 1})
        if isoSpa:
            out_filter2 = [2*start_filter]
        else:
            out_filter2 = [2*start_filter, 2*math.ceil(start_filter/4)]
            #out_filter2 = [32, 8]
        self.conv2 = build_pq_layer(out_filter1, out_filter2,#[16, 4], [32, 8],
                        p_kernel_size=kernel_sizeSpa,
                        kernel="pq_TP",
                        q_sampling_schema_in=vec[1],
                        q_sampling_schema_out=vec[2],
                        use_batch_norm=True,
                        transposed=False,
                        non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                        use_non_linearity=True,
                        p_radial_basis_type="cosine",
                        p_radial_basis_params={"num_layers": 3, "num_units": 50},
                        sub_kernel_selection_rule={"l_diff_to_out_max": 1})
        if isoSpa:
            out_filter3 = [4*start_filter]
        else:
            out_filter3 = [4*start_filter, 2*math.ceil(start_filter/4), math.ceil(start_filter/8)]
            #out_filter3 = [64, 8, 2]
        self.conv3 = build_pq_layer(out_filter2, out_filter3,#[32, 8], [64, 8, 2],
                        p_kernel_size=kernel_sizeSpa,
                        kernel="pq_TP",
                        q_sampling_schema_in=vec[2],
                        q_sampling_schema_out=vec[2],
                        use_batch_norm=True,
                        transposed=False,
                        non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                        use_non_linearity=True,
                        p_radial_basis_type="cosine",
                        p_radial_basis_params={"num_layers": 3, "num_units": 50},
                        sub_kernel_selection_rule={"l_diff_to_out_max": 1})

        self.reduction, _ = build_q_reduction_layer(out_filter3, vec[2], reduction='length_weighted_average')

        self.conv4 = build_p_layer(out_filter3, [n_class],#[64, 8, 2], [10],
                            kernel_size=kernel_sizeSpa,
                            non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                            use_non_linearity=False,
                            use_batch_norm=False,
                            transposed=False,
                            p_radial_basis_type="cosine",
                            p_radial_basis_params={"num_layers": 3, "num_units": 50})

                                    
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        x = self.conv1(x)
        #x, _, _ = self.pooling(x)
        x = self.conv2(x)
        #x, _, _ = self.pooling(x)
        x = self.conv3(x)
        #x, _, _ = self.pooling(x)
        #x = self.reduction(x)
        x = torch.mean(x, axis=2)
        x = self.conv4(x)
        return x


class Segmentation2(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, kernel_sizeSpa, vec, pool, start_filter=32, isoSpa=True, input_features=1, n_class=11):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(Segmentation2, self).__init__()
        self.pooling = pool.pooling
        if isoSpa:
            out_filter1 = [start_filter]
        else:
            out_filter1 = [start_filter, start_filter]
            out_filter1 = [16, 4]
        self.conv1 = build_pq_layer([input_features], out_filter1, #[1], [16, 4],
                p_kernel_size=kernel_sizeSpa,
                kernel="pq_TP",
                q_sampling_schema_in=vec[0],
                q_sampling_schema_out=vec[0],
                use_batch_norm=True,
                transposed=False,
                non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                use_non_linearity=True,
                p_radial_basis_type="cosine",
                p_radial_basis_params={"num_layers": 3, "num_units": 50},
                sub_kernel_selection_rule={"l_diff_to_out_max": 1})

        self.reduction, _ = build_q_reduction_layer(out_filter1, vec[0], reduction='length_weighted_average')

        if isoSpa:
            out_filter2 = [2*start_filter]
        else:
            out_filter2 = [2*start_filter, 2*start_filter, start_filter]
            out_filter2 = [32, 8]
        self.conv2 = build_p_layer(out_filter1, out_filter2,
                            kernel_size=kernel_sizeSpa,
                            non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                            use_non_linearity=True,
                            use_batch_norm=True,
                            transposed=False,
                            p_radial_basis_type="cosine",
                            p_radial_basis_params={"num_layers": 3, "num_units": 50})
        if isoSpa:
            out_filter3 = [4*start_filter]
        else:
            out_filter3 = [4*start_filter, 4*start_filter, 2*start_filter]
            out_filter3 = [64, 8, 2]
        self.conv3 = build_p_layer(out_filter2, out_filter3,
                            kernel_size=kernel_sizeSpa,
                            non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                            use_non_linearity=True,
                            use_batch_norm=True,
                            transposed=False,
                            p_radial_basis_type="cosine",
                            p_radial_basis_params={"num_layers": 3, "num_units": 50})

        self.conv4 = build_p_layer(out_filter3, [n_class],#[64, 8, 2], [10],
                            kernel_size=kernel_sizeSpa,
                            non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                            use_non_linearity=False,
                            use_batch_norm=False,
                            transposed=False,
                            p_radial_basis_type="cosine",
                            p_radial_basis_params={"num_layers": 3, "num_units": 50})

                                    
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        x = self.conv1(x)
        x = self.reduction(x)
        #x, _, _ = self.pooling(x)
        x = self.conv2(x)
        #x, _, _ = self.pooling(x)
        x = self.conv3(x)
        #x, _, _ = self.pooling(x)
        x = self.conv4(x)
        return x