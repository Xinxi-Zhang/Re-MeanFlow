import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import random
from copy import deepcopy
import dnnlib
import argparse
import json
import logging
import os
from accelerate import Accelerator
from meanflow import MeanFlow
import torchvision
from tqdm import tqdm
import wandb
import numpy as np
######edm2
import pickle
from edm2.networks_edm2 import Precond
from accelerate.utils import DistributedDataParallelKwargs
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.fid_score import calculate_frechet_distance

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
class ReflowDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.meta_data = json.load(f)

    def __len__(self):
        return len(self.meta_data["file_list"])
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.meta_data["root"], self.meta_data["file_list"][idx])
        noise_path = image_path.replace(".npy", "_noise.npy")
        label = self.meta_data["label_list"][idx]

        image = torch.from_numpy(np.load(image_path))
        noise = torch.from_numpy(np.load(noise_path))

        return image, noise, label  

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

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
    # Setup accelerator:
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=args.gradient_accumulation_steps)
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
        eval_dir = f"{experiment_dir}/eval"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
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
    # 1. Create model                                      #
    #########################################################
    # Load EDM-S
    preset = config_presets[args.model]
    net = preset['net']
    with dnnlib.util.open_url(net, verbose=True) as f:
        data = pickle.load(f)
    net = data['ema']

    model = Precond(*net.init_args, **net.init_kwargs)
    model.load_state_dict(net.state_dict(), strict=False)
    vae = data.get('encoder', None)
    vae.init(device)
    
    # setup t_emb
    del data, model.unet.emb_noise
    model.unet.emb_noise_t.load_state_dict(torch.load(args.t_emb_ckpt, map_location=device))
    model.unet.emb_noise_r.load_state_dict(torch.load(args.r_emb_ckpt, map_location=device))
    
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if accelerator.is_main_process:
        logger.info(f"EDM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume training from a checkpoint if provided and exists
    if args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint = torch.load(args.model_ckpt, map_location=torch.device(device), weights_only=False)
        init_epoch = checkpoint["epoch"] 
        if args.resume_ema:
            model.load_state_dict(checkpoint["ema"])
            ema.load_state_dict(checkpoint["ema"])
            if accelerator.is_main_process:
                logger.info("================================== Loaded pretrained EMA ==================================")
        else:
            model.load_state_dict(checkpoint["model"])
            if not args.no_resume_opt:
                opt.load_state_dict(checkpoint["opt"])
            ema.load_state_dict(checkpoint["ema"])
        train_steps = checkpoint["train_steps"]
        if accelerator.is_main_process:
            logger.info("=> loaded checkpoint (epoch {})".format(init_epoch))
        del checkpoint   
    else:
        init_epoch = 0
        train_steps = 0
        
    # Setup optimizer
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    param_list = list(model_params)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    #########################################################
    # 2. Load dataset                                       #
    #########################################################
    dataset = ReflowDataset(args.json_path)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images")

    #########################################################
    # 3. Prepare models for training
    #########################################################
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    meanflow = MeanFlow(model,
                        image_size=preset['image_size'],
                        in_channel=preset['in_channels'],
                        p_std=args.p_std,
                        p_mean=args.p_mean,
                        consistency_ratio=args.consistency_ratio,
                        jvp_api=args.jvp_api,
                        w=args.w,
                        w_pi=args.w_pi,
                        p=args.p,
                        pc=args.pc,
                        t_r_schedule=args.t_r_schedule)

    #########################################################
    # 4. Evaluation
    #########################################################
    fid_stats = np.load(args.fid_stats_path)
    real_mu = fid_stats['mu']
    real_sigma = fid_stats['sigma']
    sample_num = 50000
    
    def eval_fid(bs):
        torch.cuda.empty_cache()
        ema.eval()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.v3_dims]
        v3_model = InceptionV3([block_idx]).to(device)
        v3_model.eval()
        iterations = sample_num // (bs * accelerator.num_processes) + 1

        if accelerator.is_main_process:
            logger.info(f"Evaluating FID...")
        
        print(f"rank: {rank}, iterations: {iterations}, bs: {bs}, sample_num: {sample_num}")

        pred_list = []
        for i in (tqdm(range(iterations), desc=f"Evaluating FID") if rank == 0 else range(iterations)):
            # sample a class list of 1000 classes with bs
            class_idx = torch.randint(0, 1000, (bs,), device=device)
            samples = meanflow.sample_one_step(ema, class_idx=class_idx, device=device)
            samples = vae.decode(samples)
            samples = samples.float() / 255.0

            pred = v3_model(samples)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu()
            pred_list.append(pred)
                
        # clear gpu cache
        torch.cuda.empty_cache()
        pred_list = torch.cat(pred_list, dim=0).to(device)

        gathered = accelerator.gather(pred_list)
        if accelerator.is_main_process:
            pred_arr = gathered.cpu().numpy()
            print(f"pred_arr shape: {pred_arr.shape}")
            mu = np.mean(pred_arr, axis=0)
            sigma = np.cov(pred_arr, rowvar=False)
            fid = calculate_frechet_distance(mu, sigma, real_mu, real_sigma)
            logger.info(f"fid: {fid}")

            # save some samples
            torchvision.utils.save_image(samples, f"{eval_dir}/{train_steps:07d}.png", nrow=4, normalize=False)
            if args.wandb:
                wandb_run.log({"fid_images": wandb.Image(f"{eval_dir}/{train_steps:07d}.png", caption="fid_samples"),
                               "fid": fid})
            
            del pred_arr, mu, sigma
        
        del v3_model, pred_list, gathered
        torch.cuda.empty_cache()
        
    #########################################################
    # 5. Training
    #########################################################
    if accelerator.is_main_process:
        logger.info(f"Training for {args.iterations} iterations...")
        plot_classes = [471, 720, 114, 752, 523, 427, 937, 698]
        eval_noise = torch.randn(len(plot_classes), preset['in_channels'], preset['image_size'], preset['image_size'], device=device)        
        with torch.no_grad():
            x = meanflow.sample_one_step(ema, n=eval_noise, device=device)
            x_edm = meanflow.sample_edm(ema, n=eval_noise, device=device)
            samples = torch.cat([x, x_edm], dim=0)
            samples = vae.decode(samples)
            samples = samples.float() / 255.0
            torchvision.utils.save_image(samples, f"{samples_dir}/init.png", nrow=4, normalize=False)

            if args.wandb:
                wandb_run.log({"init": wandb.Image(f"{samples_dir}/init.png", caption="init_samples")})

    for epoch in range(init_epoch, args.epochs+1):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
            
        for x, n, y in tqdm(loader, total=len(loader)):
            with accelerator.accumulate(model):
                x = x.to(device)
                n = n.to(device)
                y = y.to(device)
                bs = x.shape[0]

                # Distance Truncation
                if args.distance_clip > 0:
                    dis = torch.mean((n-x)**2, dim=(1, 2, 3))
                    mask = dis < args.distance_clip
                    if sum(mask) == 0:
                        continue
                    x = x[mask]
                    n = n[mask]
                    y = y[mask]
                  
                if train_steps > args.stage1_iterations:
                    meanflow.w_pi = args.w_pi
                    meanflow.w_pi = 1 + (args.w_pi-1) * torch.rand(1, device=device).item()  # Random value between 1 and args.w_pi
                    meanflow.w = max(1.0, meanflow.w_pi-1)
                    args.grad_clip = 1.0 # clip grad norm to further stabilize training on the CFG flow
                else:
                    meanflow.w_pi = 1.0

                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    loss_dict = meanflow.loss_edm(x, y, noise=n)

                # manage grad and update model
                opt.zero_grad()
                accelerator.backward(loss_dict["loss"])
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                
            if accelerator.sync_gradients:
                update_ema(ema, model, decay=args.ema_rate)
                train_steps += 1

            # Log loss values:
            if accelerator.is_main_process:
            
                if train_steps % args.log_every == 0:
                    log_dict = {"steps": train_steps, "epoch": epoch, "w_pi": meanflow.w_pi, "loss": loss_dict["loss"].item()}
                    logger.info(f"Steps: {train_steps}, Epoch: {epoch}, w_pi: {meanflow.w_pi}, loss: {loss_dict['loss'].item()}")

                    if args.wandb:
                        wandb_run.log(log_dict)

            # Evaluate FID
            if train_steps % args.eval_every == 0:
                accelerator.wait_for_everyone()
                with torch.no_grad():
                    eval_fid(bs)

            # Plot samples
            if accelerator.is_main_process:
                if train_steps % args.plot_every == 0:
                    # sample from train model and ema model
                    with torch.no_grad():
                        x_train = meanflow.sample_one_step(model, n=eval_noise, device=device)
                        x_ema = meanflow.sample_one_step(ema, n=eval_noise, device=device)
                        x = torch.cat([x_train, x_ema], dim=0)
                        
                        samples = vae.decode(x)
                        samples = samples.float() / 255.0
                        torchvision.utils.save_image(samples, f"{samples_dir}/steps_{train_steps:07d}.png", nrow=4, normalize=False)
                        if args.wandb:
                            wandb_run.log({"samples": wandb.Image(f"{samples_dir}/steps_{train_steps:07d}.png", caption="samples"), "steps": train_steps})
                if train_steps % args.ckpt_every == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "epoch": epoch,
                        "train_steps": train_steps,
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    # data and dir args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--json_path", type=str, default="dataset/512/reflow/dataset.json")
    parser.add_argument("--fid_stats_path", type=str, default="tools/fid_stats/imagenet512_stats.npz")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--exp_group", type=str, default="meanflow")
    parser.add_argument("--results_dir", type=str, default="R/512")

    # model args
    parser.add_argument("--model", type=str, default="edm2-img512-s-autog-fid")
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--t_emb_ckpt", type=str, default="R/512/time_emb_512/checkpoints/emb_t_0010000.pt")
    parser.add_argument("--r_emb_ckpt", type=str, default="R/512/time_emb_512/checkpoints/emb_r_0010000.pt")

    # meanflow args
    parser.add_argument("--t_r_schedule", type=str, choices=["uniform", "re2_truncate"], default="re2_truncate")
    parser.add_argument("--p_std", type=float, default=1.0)
    parser.add_argument("--p_mean", type=float, default=-0.4)
    parser.add_argument("--consistency_ratio", type=float, default=0.25)
    parser.add_argument("--jvp_api", type=str, default="autograd")
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--pc", type=float, default=1e-3)
    parser.add_argument("--stage1_iterations", type=int, default=50000)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--w_pi", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, choices=["adaptive_loss_weight"], default="adaptive_loss_weight")
    parser.add_argument("--distance_clip", type=float, default=0.0)

    # optimization args
    parser.add_argument("--optimizer", type=str, choices=["adamw", "adamw"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)

    # training args
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--plot_every", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_ema", action="store_true")
    
    # fid args
    parser.add_argument("--v3_dims", type=int, default=2048)

    args = parser.parse_args()
    main(args)