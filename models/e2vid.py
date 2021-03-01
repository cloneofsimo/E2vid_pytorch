<<<<<<< HEAD
import torch
import torch.nn as nn
from torch import optim

from omegaconf.dictconfig import DictConfig
from .conv_modules import *
import pytorch_lightning as pl





import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def show_img(img):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Img")
    plt.imshow(
        np.transpose(
            vutils.make_grid(img, padding=2, normalize=True, range=(-1, 1)).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()


class e2vid(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.core = unetCore(cfg)
        self.criterion = nn.MSELoss()

    def forward(self, x, y=None):
        yh = self.core(x)
        if y is not None:
            loss = self.criterion(yh, y)
            return yh, loss
        else:
            return yh

    def training_step(self, batch, batch_idx):

        x, y = batch
        _, loss = self.forward(x, y)

        tensorboard_log = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yh, loss = self.forward(x, y)
        
        

        tensorboard_log = {"val_loss": loss}

        return {"val_loss": loss, "log": tensorboard_log, "yh" : yh, "y" : y}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        show_img(outputs[-1]["yh"])
        show_img(outputs[-1]["y"])
        tensorboard_log = {"val_loss": loss}
        return {"val_loss": loss, "log": tensorboard_log}

    def configure_optimizers(self):
        if self.cfg.train.optim.type == "adam":
            return optim.Adam(
                self.parameters(),
                lr=self.cfg.train.optim.lr,
                betas=self.cfg.train.optim.betas,
                weight_decay=self.cfg.train.optim.weight_decay,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

=======
import torch
import torch.nn as nn
from torch import optim

from omegaconf.dictconfig import DictConfig
from .conv_modules import *
import pytorch_lightning as pl


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
        self.ups = nn.Sequential(conv3(4 * N, N, cfg), conv3(2 * N, 1, cfg))
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


class e2vid(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.core = unetCore(cfg)
        self.criterion = nn.MSELoss()

    def forward(self, x, y=None):
        yh = self.core(x)
        if y is not None:
            loss = self.criterion(yh, y)
            return yh, loss
        else:
            return yh

    def training_step(self, batch, batch_idx):

        x, y = batch
        _, loss = self.forward(x, y)

        tensorboard_log = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.forward(x, y)

        tensorboard_log = {"val_loss": loss}

        return {"val_loss": loss, "log": tensorboard_log}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_log = {"val_loss": loss}
        return {"val_loss": loss, "log": tensorboard_log}

    def configure_optimizers(self):
        if self.cfg.train.optim.type == "adam":
            return optim.Adam(
                self.parameters(),
                lr=self.cfg.train.optim.lr,
                betas=self.cfg.train.optim.betas,
                weight_decay=self.cfg.train.optim.weight_decay,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

>>>>>>> 069632990ad6e4e634d4d589f374c26a6d365e72
