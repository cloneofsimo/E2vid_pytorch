<<<<<<< HEAD
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import torchvision
import PIL


import pandas as pd
from glob import glob
from random import randint
from tqdm import tqdm

import os
import hydra
from omegaconf.dictconfig import DictConfig

from .utils import show_img

from .utils import extract_event_tensor, generate_index_table, generate_img_table


def hopath(path):
    return os.path.join(hydra.utils.get_original_cwd(), path)


class e2tensor_dataset(Dataset):
    def __init__(self, cfg, dpath, R1=0, R2=230):
        super().__init__()
        self.cfg = cfg
        self.dpath = dpath
        self.img_table = generate_img_table(datapath=dpath)[R1:R2]
        self.time_table = generate_index_table(datapath=dpath)
        self.require_y = True

        self.tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )

        #@self.ets = [extract_event_tensor()]

    def __len__(self):
        return len(self.img_table)

    def __getitem__(self, idx):
        event_tensor = extract_event_tensor(
            time=self.img_table[idx][0], event_index=self.time_table
        )
        #show_img(event_tensor)
        print(event_tensor.shape)
        if self.require_y:
            img = PIL.Image.open(os.path.join(self.dpath, self.img_table[idx][1]))
            return event_tensor / 0.7, self.tf(img)
        else:
            return event_tensor




class e2tensor_datamodule(pl.LightningDataModule):
    def __init__(self, cfg, dpath):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers

        self.train_dataset = e2tensor_dataset(cfg, dpath, R1=0, R2=230)
        self.val_dataset = e2tensor_dataset(cfg, dpath, R1=230, R2=300)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle= True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

=======
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import torchvision
import PIL


import pandas as pd
from glob import glob
from random import randint
from tqdm import tqdm

import os
import hydra
from omegaconf.dictconfig import DictConfig


from .utils import extract_event_tensor, generate_index_table, generate_img_table


def hopath(path):
    return os.path.join(hydra.utils.get_original_cwd(), path)


class e2tensor_dataset(Dataset):
    def __init__(self, cfg, dpath, R1=0, R2=230):
        super().__init__()
        self.cfg = cfg
        self.dpath = dpath
        self.img_table = generate_img_table(datapath=dpath)[R1:R2]
        self.time_table = generate_index_table(datapath=dpath)
        self.require_y = True

        self.tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )

    def __len__(self):
        return len(self.img_table)

    def __getitem__(self, idx):
        event_tensor = extract_event_tensor(
            time=self.img_table[idx][0], event_index=self.time_table
        )
        if self.require_y:
            img = PIL.Image.open(os.path.join(self.dpath, self.img_table[idx][1]))
            return event_tensor, self.tf(img)
        else:
            return event_tensor


class e2tensor_datamodule(pl.LightningDataModule):
    def __init__(self, cfg, dpath):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers

        self.train_dataset = e2tensor_dataset(cfg, dpath, R1=0, R2=230)
        self.val_dataset = e2tensor_dataset(cfg, dpath, R1=230, R2=300)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

>>>>>>> 069632990ad6e4e634d4d589f374c26a6d365e72
