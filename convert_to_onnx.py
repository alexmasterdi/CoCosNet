import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import torch.nn as nn

import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import torch.onnx.symbolic_registry as sym_registry


opt = TestOptions().parse()


def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass

        for child in item.children():
            remove_all_spectral_norm(child)

    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)

    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)


model = Pix2PixModel(opt)
model.eval()

remove_all_spectral_norm(model)

ref_image = torch.randn(1, 3, 256, 256)
real_image = torch.randn(1, 3, 256, 256)
input_sem = torch.randn(1, 151, 256, 256)
ref_sem = torch.randn(1, 151, 256, 256)
warp_out = torch.randn(1, 154, 256, 256)


def own_var(input, dim, keepdim=False, unbiased=True, out=None):
    E = torch.mean(input, dim)
    E_up = torch.mean(torch.pow(input, 2), dim)
    Var = E_up - torch.pow(E, 2)
    shape = input.shape
    print(shape[dim])
    return Var * (shape[dim]/(shape[dim] - 1))


def symbolic_range_tensor(g, start, end, step):
    return g.op("Range", start, end, step)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('custom_ops::range_tensor', symbolic_range_tensor, 9)
torch.onnx.export(model.net['netCorr'], (ref_image, real_image, input_sem, ref_sem), "Corr.onnx", opset_version=11, enable_onnx_checker=True)
