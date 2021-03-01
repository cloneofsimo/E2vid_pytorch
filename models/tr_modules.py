import torch
import torch.nn as nn


act_types = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "lelu": nn.LeakyReLU(negative_slope=0.2),
}