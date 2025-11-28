
import torch
import argparse
import json
import logging
import os
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
######edm2
import pickle
from edm2.edm2_og import Precond
import dnnlib
from accelerate import Accelerator
import random
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

class ReflowDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.meta_data = json.load(f)
            self.image_root = self.meta_data['root']
            os.makedirs(self.image_root, exist_ok=True)

    def __len__(self):
        return len(self.meta_data["file_list"])
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_root, self.meta_data["file_list"][idx])
        label = self.meta_data["label_list"][idx]

        # to check if the image is already generated
        exist = not os.path.exists(image_path)

        return image_path, label, exist

def build_edm_labels(c: torch.Tensor, num_classes: int = 1000, device: torch.device = None):
    labels = torch.eye(num_classes, device=device)
    zero = torch.zeros((1, num_classes), device=device)
    labels = torch.cat([labels, zero], dim=0)[c].contiguous()

    return labels

def edm_forward(model, gnet, c, w, n, num_steps, num_classes, device):
    def denoise(x, t, labels):
        Dx = model(x, t, labels).to(x.dtype)
        ref_Dx = gnet(x, t, labels).to(x.dtype)

        return ref_Dx.lerp(Dx, w)

    sigma_max = 80
    sigma_min = 0.002
    rho = 7

    step_indices = torch.arange(num_steps, dtype=n.dtype, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    labels = build_edm_labels(c, num_classes, device)
    x_next = n * t_steps[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            t_hat = t_cur
            x_hat = x_cur

            # Euler step.
            d_cur = (x_hat - denoise(x_hat, t_hat, labels)) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < num_steps - 1:
                d_prime = (x_next - denoise(x_next, t_next, labels)) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
    return x_next

edm2_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'
config_presets = {
    'edm2-img64-s-autog-fid':          dnnlib.EasyDict(net=f'{edm2_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{edm2_root}/edm2-img64-xs-0134217-0.110.pkl',         
    guidance=1.70, image_size=64, in_channels=3, sampling_steps=32, class_num=1000),
    'edm2-img512-s-autog-fid':         dnnlib.EasyDict(net=f'{edm2_root}/edm2-img512-s-2147483-0.070.pkl',gnet=f'{edm2_root}/edm2-img512-xs-0134217-0.125.pkl', guidance=2.10,
    image_size=64, in_channels=4, sampling_steps=32, class_num=1000), # fid = 1.34
}

def main(args):
    #########################################################
    # 0. Setup 
    #########################################################0
    # set the seed
    accelerator = Accelerator()
    device = accelerator.device

    # set seed
    rank = device.index if device.index is not None else 0
    torch.manual_seed(args.global_seed + rank)
    np.random.seed(args.global_seed + rank)
    random.seed(args.global_seed + rank)

    if accelerator.is_main_process:
        # check dirs
        logger = create_logger(args.data_path)

        if args.wandb:
            wandb_run = wandb.init(
                entity="your entity",
                project="Re-MeanFlow",
                name=f"reflow_{args.resolution}",
                group="RE_EDM",
                config=vars(args),
            )

    #########################################################
    # 1. Load model
    #########################################################
    # Create model:
    preset = config_presets[args.model]
    net = preset['net']
    with dnnlib.util.open_url(net, verbose=True) as f:
        data = pickle.load(f)
    
    net = data['ema']

    model = Precond(*net.init_args, **net.init_kwargs)
    model.load_state_dict(net.state_dict(), strict=True)
    model.to(device)
    model.eval()

    # Load gnet for auto-guidance
    if 'gnet' in preset:
        gnet = preset['gnet']
        with dnnlib.util.open_url(gnet, verbose=True) as f:
            data = pickle.load(f)
        gnet = data['ema']
        gnet = gnet.to(device)
        gnet.eval()
    else:
        gnet = net
    
    # Load VAE 
    vae = data.get('encoder', None)
    vae.init(device)

    ds = ReflowDataset(args.json_path)
    loader = DataLoader(ds, batch_size=args.bs, num_workers=4, drop_last=False)

    loader = accelerator.prepare(loader)

    epoch = 0
    # sample the data
    for batch in tqdm(loader, desc="Loading data") if rank == 0 else loader:
        image_path, label, exist = batch

        # if no image to generate, skip
        if exist.sum() == 0:
            continue
        
        image_path = [p for p,e in zip(image_path, exist) if e == 1]
        label = label[exist.bool()]
        
        latent_noise = torch.randn(label.shape[0], preset['in_channels'], preset['image_size'], preset['image_size'], device=device)
        latent =  edm_forward(model, gnet, label, preset['guidance'], latent_noise, preset['sampling_steps'], preset['class_num'], device)

        for latent_path, idx in zip(image_path, range(len(image_path))):
            np.save(latent_path, latent[idx].cpu().numpy())
            noise_path = latent_path.replace('.npy', '_noise.npy')
            np.save(noise_path, latent_noise[idx].cpu().numpy())
        
        # save samples every 100 epochs
        if epoch % 100 == 0 and rank == 0:
            samples = latent[:16]
            samples = vae.decode(samples)
            samples = samples.float() / 255.0
            torchvision.utils.save_image(samples, f"{args.data_path}/samples.png", nrow=4, normalize=False)
            
            noise = latent_noise[:16]
            torchvision.utils.save_image(noise, f"{args.data_path}/noise.png", nrow=4, normalize=True, value_range=(-1, 1))

            if args.wandb:
                wandb_log = {}
                wandb_log[f"samples"] = wandb.Image(f"{args.data_path}/samples.png")
                wandb_log[f"noise"] = wandb.Image(f"{args.data_path}/noise.png")
                wandb_run.log(wandb_log)
        
        epoch += 1
   
    if rank == 0:
        logger.info("Done!")
        if args.wandb:
            wandb_run.finish()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument("--data_path", type=str, default="dataset/512/reflow")
    parser.add_argument("--json_path", type=str, default="dataset/512/reflow/dataset.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--global_seed", type=int, default=0)

    # model args
    parser.add_argument("--model", type=str, default="edm2-img512-s-autog-fid")
    parser.add_argument("--num_classes", type=int, default=1000)

    args = parser.parse_args()
    with torch.no_grad():
        main(args)