import os
import platform
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
import torch


from models.e2vid import e2vid
from dataset.bin_data import e2tensor_datamodule
from dataset.utils import extract_event_tensor, generate_index_table, generate_img_table, show_img, save_img


class interpolation_module():
    def __init__(self, dpath, R1, R2):
        self.dpath = dpath
        self.img_table = generate_img_table(datapath=dpath)[R1:R2]
        self.time_table = generate_index_table(datapath=dpath)
        self.require_y = True

        self.tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
    
    def event_tensor_of_time(self, time = 0.2):
        event_tensor = extract_event_tensor(
            time= time, event_index=self.time_table
        )
        return event_tensor.unsqueeze(dim = 0)

def hopath(path):
    return os.path.join(hydra.utils.get_original_cwd(), path)

@hydra.main(config_name="cnn_inf_config")
def main(cfg: DictConfig):
    model = e2vid(cfg)

    checkpoint_path = hopath(cfg.inference.checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    device = "cuda:0"
    model.to(device)

    time_dataset = interpolation_module(hopath(cfg.dataset.dir), 0, 280)

    SPC_TIME = 10.654906000
    IMG = model(time_dataset.event_tensor_of_time(time = SPC_TIME).to(device)).detach()
    show_img(IMG)
    TIME = 0
    while TIME < 4:
        TIME += 1/120
        IMG = model(time_dataset.event_tensor_of_time(time = TIME).to(device)).detach()
        save_img(IMG, path = hopath(f"results/120fps/{TIME}.png"))

if __name__ == "__main__":
    torch.no_grad()
    main()
