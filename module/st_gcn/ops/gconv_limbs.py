import torch
import torch.nn as nn
import numpy as np
from .graph_limbs import LimbsGraph
from .graph_adplimbs import adplimbsGraph

class ConvTemporalGraphicalLimbs(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 kernel_size=3,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 limbs_graph=None):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, limb_A):
        assert limb_A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, limb_A))
        return x.contiguous(), limb_A