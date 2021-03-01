<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from bisect import bisect
import torch


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

def save_img(img, path):
    vutils.save_image(img, normalize= True, range=(-1, 1),
                fp = path)


def generate_img_table(datapath="./data/urban"):
    with open(os.path.join(datapath, "images.txt"), "r") as FIN:
        data = FIN.read().strip().split("\n")
        data = [(float(t), dirval) for (t, dirval) in [r.split() for r in data]]
        return data


def generate_index_table(datapath="./data/urban"):
    time_ranges = [0]
    time_idx = [0]
    with open(os.path.join(datapath, "events.txt")) as FIN:
        data = FIN.read().strip().split("\n")
        for idx, eve in enumerate(data):
            T, _, _, _ = eve.split()
            if T != time_ranges[-1]:
                time_ranges.append(float(T))
                time_idx.append(idx)

    return time_ranges[1:], time_idx, data



def extract_event_tensor(
    time=2.0, n_bin=4, N=25000, datapath="./data/urban", event_index=None, H=180, W=240
):

    if event_index is None:
        event_index = generate_index_table(datapath)

    time_ranges, time_idx, data = event_index

    # print(time_ranges, time_idx)

    here = bisect(time_ranges, time)
    T_0 = float(data[here].split()[0])
    T_m1 = float(data[here - N * n_bin].split()[0])
    nC = (n_bin - 1) / (T_0 - T_m1)

    temporal_bins = []

    for bidx in range(n_bin):
        E = torch.zeros(H, W)
        T_h, _, _, _ = data[here - N * bidx].split()
        T_h = float(T_h)
        T_h = nC * (T_h - T_0)

        for idx in range(here - N * (bidx + 1), here - N * bidx):
            T, x, y, pol = data[idx].split()
            x, y = int(x), int(y)
            T = float(T)
            T_n = nC * (T - T_0)
            a = (2 * float(pol) - 1) * max(1 - abs(T_h - T_n), 0)
            E[y, x] += a

        temporal_bins.append(E.unsqueeze(dim=0))

    return torch.cat(temporal_bins, dim=0)


if __name__ == "__main__":
    generate_img_table()
    x = extract_event_tensor()[1:2, :, :]
    print(x)
    print(x.std())
    x = (x - x.mean()) / (x.std() + 1e-9)

    print(torch.histc(x, 100, -10, 10))

    print(x.shape)
    show_img(x)
=======
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from bisect import bisect
import torch


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


def generate_img_table(datapath="./data/urban"):
    with open(os.path.join(datapath, "images.txt"), "r") as FIN:
        data = FIN.read().strip().split("\n")
        data = [(float(t), dirval) for (t, dirval) in [r.split() for r in data]]
        return data


def generate_index_table(datapath="./data/urban"):
    time_ranges = [0]
    time_idx = [0]
    with open(os.path.join(datapath, "events.txt")) as FIN:
        data = FIN.read().strip().split("\n")
        for idx, eve in enumerate(data):
            T, _, _, _ = eve.split()
            if T != time_ranges[-1]:
                time_ranges.append(float(T))
                time_idx.append(idx)

    return time_ranges[1:], time_idx, data


def extract_event_tensor(
    time=2.0, n_bin=4, N=25000, datapath="./data/urban", event_index=None, H=180, W=240
):

    if event_index is None:
        event_index = generate_index_table(datapath)

    time_ranges, time_idx, data = event_index

    # print(time_ranges, time_idx)

    here = bisect(time_ranges, time)
    T_0 = float(data[here].split()[0])
    T_m1 = float(data[here - N * n_bin].split()[0])
    nC = (n_bin - 1) / (T_0 - T_m1)

    temporal_bins = []

    for bidx in range(n_bin):
        E = torch.zeros(H, W)
        T_h, _, _, _ = data[here - N * bidx].split()
        T_h = float(T_h)
        T_h = nC * (T_h - T_0)

        for idx in range(here - N * (bidx + 1), here - N * bidx):
            T, x, y, pol = data[idx].split()
            x, y = int(x), int(y)
            T = float(T)
            T_n = nC * (T - T_0)
            a = (2 * float(pol) - 1) * max(1 - abs(T_h - T_n), 0)
            E[y, x] += a

        temporal_bins.append(E.unsqueeze(dim=0))

    return torch.cat(temporal_bins, dim=0)


if __name__ == "__main__":
    generate_img_table()
    x = extract_event_tensor()[0:1, :, :]
    print(x)
    print(x.std())
    x = (x - x.mean()) / (x.std() + 1e-9)

    print(torch.histc(x, 100, -10, 10))

    print(x.shape)
    show_img(x)
>>>>>>> 069632990ad6e4e634d4d589f374c26a6d365e72
