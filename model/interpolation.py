import torch


class Interpolation(torch.nn.Module):
    def __init__(self, shellSampling, gridSampling, conv_name):
        """Initialization.
        Args:
            shellSampling (:obj:`sampling.ShellSampling`): Input sampling scheme
            gridSampling (:obj:`sampling.Sampling`): Interpolation grid scheme
        """
        super(Interpolation, self).__init__()
        self.shellSampling = shellSampling
        self.gridSampling = gridSampling
        self.S = len(shellSampling.sampling)
        SH2S = self.gridSampling.sampling.SH2S
        C_grid = SH2S.shape[0]
        self.conv_name = conv_name
        for i, sampling in enumerate(self.shellSampling.sampling):
            S2SH = sampling.S2SH
            C = S2SH.shape[1]
            assert C <= C_grid
            if conv_name in ['spatial_sh']:
                self.register_buffer(f'sampling2sampling_{i}', torch.Tensor(S2SH))
                self.V = S2SH.shape[1]
            else:
                self.register_buffer(f'sampling2sampling_{i}', torch.Tensor(S2SH.dot(SH2S[:C])))
                self.V = SH2S.shape[1]

    def sampling2sampling(self, i):
        return self.__getattr__('sampling2sampling_'+str(i)) # C x V_i

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V' x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: interpolated input [B x in_channels*S x V x X x Y x Z]
        """
        B, F_in, _, X, Y, Z = x.shape
        y = x.new_zeros(B, F_in*self.S, self.V, X, Y, Z) # B x in_channels' x V x X x Y x Z
        for i in range(self.S):
            x_shell = x[:, :, self.shellSampling.shell_inverse == i] # B x in_channels x V_i x X x Y x Z
            y[:, i*F_in:(i+1)*F_in] = torch.einsum('ijklmn,kp->ijplmn', x_shell, self.sampling2sampling(i)) # B x in_channels x V x X x Y x Z
        if self.conv_name in ['spatial_vec', 'spatial_sh']:
            y = y.reshape(B, -1, 1, X, Y, Z).contiguous()
        return y
