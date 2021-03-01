import torch
import torch.nn as nn


act_types = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "lelu": nn.LeakyReLU(negative_slope=0.2),
}


class conv3(nn.Module):
    def __init__(self, ic, oc, cfg):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
            nn.Conv2d(oc, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
            nn.Conv2d(oc, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class unetCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg.model.conv.n_channels
        N = 128
        self.downs = nn.Sequential(
            conv3(M, N, cfg), conv3(N, 2 * N, cfg), conv3(2 * N, 2 * N, cfg),
        )
        self.maxpools = nn.Sequential(
            nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2),
        )
        self.ups = nn.Sequential(conv3(4 * N, N, cfg), nn.Conv2d(2*N, 1, 3, 1, 1))
        self.upconvs = nn.Sequential(
            nn.ConvTranspose2d(2 * N, 2 * N, 4, 2, 1),
            nn.ConvTranspose2d(N, N, 4, 2, 1),
        )
        self.last = nn.Tanh()

    def forward(self, x):

        d0 = self.downs[0](x)
        x = self.maxpools[0](d0)
        d1 = self.downs[1](x)
        x = self.maxpools[1](d1)

        x = self.downs[2](x)

        u1 = self.upconvs[0](x)
        x = torch.cat([u1, d1], dim=1)
        u0 = self.upconvs[1](self.ups[0](x))
        x = torch.cat([u0, d0], dim=1)

        x = self.ups[1](x)

        return self.last(x)
