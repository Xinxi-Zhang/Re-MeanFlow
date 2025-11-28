import torch
import numpy as np
import random
import dnnlib
import argparse
import logging
import os
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import torchvision
import pickle
from edm2.networks_edm2 import Precond, mp_sum
from meanflow import MeanFlow

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

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

edm2_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'
config_presets = {
    'edm2-img64-s-autog-fid':          dnnlib.EasyDict(net=f'{edm2_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{edm2_root}/edm2-img64-xs-0134217-0.110.pkl',         
    guidance=1.70, image_size=64, in_channels=3, sampling_steps=32, class_num=1000),
    'edm2-img512-s-autog-fid':         dnnlib.EasyDict(net=f'{edm2_root}/edm2-img512-s-2147483-0.070.pkl',gnet=f'{edm2_root}/edm2-img512-xs-0134217-0.125.pkl', guidance=2.10,
    image_size=64, in_channels=4, sampling_steps=32, class_num=1000), # fid = 1.34
}
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    #########################################################
    # 0. Setup 
    #########################################################
    accelerator = Accelerator()
    device = accelerator.device

    # set seed
    rank = device.index if device.index is not None else 0
    torch.manual_seed(args.global_seed + rank)
    np.random.seed(args.global_seed + rank)
    random.seed(args.global_seed + rank)

    # Setup an experiment folder and wandb:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_dir = f"{args.results_dir}/{args.exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        samples_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    
        # wandb
        if args.wandb:
            wandb_run = wandb.init(
                entity="your entity",
                project="Re-MeanFlow",
                name=args.exp_name,
                group=args.exp_group,
                config=vars(args),
            )

    #########################################################
    # 1. Create model
    #########################################################
    # Load EDM2 model
    preset = config_presets[args.model]
    net = preset['net']
    
    logger.info(f"Loading EDM2 model from {args.model}")
    with dnnlib.util.open_url(net, verbose=True) as f:
        data = pickle.load(f)
    model = data['ema']
    vae = data.get('encoder', None)
    vae.init(device)

    net = Precond(*model.init_args, **model.init_kwargs)
    net.load_state_dict(model.state_dict(), strict=False)
    net.to(device)
    del model
    
    emb_ori = net.unet.emb_noise
    fourier = net.unet.emb_fourier

    # Initialize New Time Embedding.
    net.unet.emb_noise_t.load_state_dict(emb_ori.state_dict())
    net.unet.emb_noise_r.load_state_dict(emb_ori.state_dict())
    emb_t, emb_r = net.unet.emb_noise_t, net.unet.emb_noise_r

    # turned off model.grad
    requires_grad(net, False)
    net.eval()

    # Load optimizer
    requires_grad(emb_t, True)
    requires_grad(emb_r, True)

    param_list = list(emb_t.parameters()) + list(emb_r.parameters())
    if args.optimizer == "adam":
        opt = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    emb_t, emb_r, opt = accelerator.prepare(emb_t, emb_r, opt)
    bs = args.batch_size

    # set sigma range, this is equivelent to [0.002, 80] in EDM2.
    sigma_min = 1/501
    sigma_max = 80/81

    # set eval noise and class
    plot_classes = [471, 720, 114, 752]
    eval_noise = torch.randn(len(plot_classes), preset['in_channels'], preset['image_size'], preset['image_size'], device=device)

    # sample using original embedding with Euler method (100 steps)
    meanflow = MeanFlow(net)
    with torch.no_grad():
        samples = meanflow.sample_edm(net, n=eval_noise, class_idx=plot_classes, device=device, og=True)
        samples = vae.decode(samples)
    samples = samples.float() / 255.0
    torchvision.utils.save_image(samples, f"{samples_dir}/init.png", nrow=4, normalize=False)
    if args.wandb:
        wandb_run.log({"init": wandb.Image(f"{samples_dir}/init.png", caption="init_samples")})
    #########################################################
    # 2. Training loop
    #########################################################
    for epoch in tqdm(range(args.epochs+1)):

        # Samples t,r uniformly from [0,1)
        t = torch.rand(bs, device=device)  
        t = t.clamp(sigma_min, sigma_max)

        r = torch.rand(bs, device=device)  
        r = r.clamp(sigma_min, sigma_max)

        # Ensure t >= r
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        
        # compute new embedding
        t = t.view(-1, 1, 1, 1).flatten()
        r = r.view(-1, 1, 1, 1).flatten()
        emb = mp_sum(emb_t(fourier(t)), emb_r(fourier(r)), t=0.5)

        with torch.no_grad():
            # original embedding
            sigma = t / (1-t)
            c_noise = sigma.flatten()
            c_noise = c_noise.log() / 4
            emb_target = emb_ori(fourier(c_noise))
        
        error = emb - emb_target
        loss = torch.mean(error ** 2)
        accelerator.backward(loss)

        opt.step()
        opt.zero_grad()

        if epoch % args.ckpt_every == 0 and accelerator.is_main_process:
            logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
            
            if hasattr(emb_t, "module"):
                torch.save(emb_t.module.state_dict(), f"{checkpoint_dir}/emb_t_{epoch:07d}.pt")
                torch.save(emb_r.module.state_dict(), f"{checkpoint_dir}/emb_r_{epoch:07d}.pt")
            else:
                torch.save(emb_t.state_dict(), f"{checkpoint_dir}/emb_t_{epoch:07d}.pt")
                torch.save(emb_r.state_dict(), f"{checkpoint_dir}/emb_r_{epoch:07d}.pt")
        
            samples = meanflow.sample_edm(net, n=eval_noise, class_idx=plot_classes, device=device)
            samples = vae.decode(samples)
            samples = samples.float() / 255.0
            torchvision.utils.save_image(samples, f"{samples_dir}/init_{epoch:07d}.png", nrow=4, normalize=False)
            if args.wandb:
                wandb_run.log({"samples": wandb.Image(f"{samples_dir}/{epoch:07d}.png")})
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # data and dir args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--exp_name", type=str, default="512_time_embedding")
    parser.add_argument("--exp_group", type=str, default="time_emb")
    parser.add_argument("--results_dir", type=str, default="R/64")
    parser.add_argument("--eval", action="store_true", default=False)

    # model args
    parser.add_argument("--model", type=str, default="edm2-img512-s-autog-fid")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--in_channels", type=int, default=4)

    # optimization args
    parser.add_argument("--optimizer", type=str, choices=["adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)

    # training args
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--ckpt_every", type=int, default=1000)

    args = parser.parse_args()
    if not args.eval:
        main(args)
    else:
        eval(args)