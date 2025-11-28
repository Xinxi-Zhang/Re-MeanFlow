<h1 align="center"> Straighter and Faster: <br> Efficient One-Step Generative Modeling 
via Meanflow on Rectified Trajectories
</h1>

<div align="center">
    <a href="https://xinxi-zhang.github.io/WEB_XINXI/" target="_blank">Xinxi&nbsp;Zhang</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://scholar.google.com/citations?user=XUsD3_kAAAAJ&amp;hl" target="_blank">Shiwei&nbsp;Tan</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://scholar.google.com/citations?user=SUuo7U4AAAAJ&amp;hl=en" target="_blank">Quang&nbsp;Nguyen</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://quandao10.github.io/" target="_blank">Quan&nbsp;Dao</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://phymhan.github.io/" target="_blank">Ligong&nbsp;Han</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://hexiaoxiao-cs.github.io/" target="_blank">Xiaoxiao&nbsp;He</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://scholar.google.com/citations?user=y3st15YAAAAJ&amp;hl=en" target="_blank">Tunyu&nbsp;Zhang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://openreview.net/profile?id=~Alen_Mrdovic1" target="_blank">Alen&nbsp;Mrdovic</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
    <a href="https://people.cs.rutgers.edu/~dnm/" target="_blank">Dimitris&nbsp;Metaxas</a><sup>1</sup>
  <br>
  <sup>1</sup> Rutgers University &emsp; <sup>2</sup> Red Hat &emsp; <br>
  <sup>*</sup>Equal contribution &emsp; <br>
</div>

<h3 align="center">[<a href="http://arxiv.org/abs/todo">arXiv</a>]</h3>
<br>

<div align="center">
<img width="800" alt="Image" src="docs/header.jpg" />
</div>
<b>Summary</b>: Re-MeanFlow enables efficient one-step generative modeling by learning mean velocities along rectified trajectories, achieving state-of-the-art one-step FID of 3.03 on ImageNet 512.  <br> <br>

We provide a minimalist codebase for Imagenet 64 (pixel space) and 512 (latent space) in this repo, initialized from [EDM2-S](https://github.com/NVlabs/edm2). For simplicity, we will only include 512×512 examples when they are nearly identical to the 64×64 setup.

### 1. Environment setup & Requirement

```bash
conda env create -f environment.yml
conda activate remeanflow
```

### 2. Dataset (Optional)
Download the ImageNet training data and extract the images to:
`[YOUR_DATA_PATH]/imagenet/raw/train`.

This step is optional for Re-MeanFlow (it does not use real images directly), but is required if you want to compute FID statistics (we provide pre-compute stats under`tools/fid_stats`).
```bash
python3 tools/download_dataset.py --path [YOUR_DATA_PATH]/imagenet \
    --url https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```

Preprocess:
```bash
python3 dataset_tools.py convert --source=[YOUR_DATA_PATH]/imagenet/raw/train \
    --dest=[YOUR_DATA_PATH]/imagenet/512/real --resolution=512x512 --transform=center-crop-dhariwal
```

### 3. Generate Reflow Couplings
We first create a JSON file listing all images to be synthesized. This can be done either by (i) sampling classes at random or (ii) matching the ImageNet class distribution using the `dataset.json` from step 2. In the paper, we use the latter.

```bash
# (i) Random classes
python3 tools/dataset_tools.py make-reflow-index \
    --dataset-root=[YOUR_DATA_PATH]/dataset/512/real \
    --dest=[YOUR_DATA_PATH]/512/reflow/dataset.json \
    --mode=random \
    --num-passes=4  # Reflow set size = 4× ImageNet

# (ii) Same class distribution as ImageNet
python3 tools/dataset_tools.py make-reflow-index \
    --dataset-root=[YOUR_DATA_PATH]/dataset/512/real \
    --dest=[YOUR_DATA_PATH]/dataset/512/reflow/dataset.json \
    --mode=imagenet \
    --num-passes=4 \
    --json-path=tools/dataset.json  # Automatically generated in step 2; also provided under tools/.
```

Then we can sampling the couplings using this json:
```bash
accelerate launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port 9527 reflow.py \
    --data_path [YOUR_DATA_PATH]/512/reflow \
    --json_path [YOUR_DATA_PATH]/512/reflow/dataset.json \ # the json generated from last step
    --model edm2-img512-s-autog-fid \ # change it to "edm2-img64-s-autog-fid" for 64x64
    --bs 1024 \
    --global_seed 0 \
```

### 4. New time embedding
As discussed in the ablations and appendix, properly initializing and replacing the original time embedding is crucial. We initialized the new time embedding described in the paper as follows:
```bash
export CUDA_VISIBLE_DEVICES=0
accelerate launch init_time_emb.py \
    --exp_name time_emb_512 \
    --results_dir R/512 \
    --model edm2-img512-s-autog-fid \ # change it to "edm2-img64-s-autog-fid" for 64x64
    --epochs 10000 \
    --batch_size 1024 \
    --global_seed 0 \
    --ckpt_every 5000 \
    #--wandb
```
This Process should only take 1 or 2 minutes.

### 5. Train Re-Meanflow
As described in the paper, we train Re-MeanFlow on rectified trajectories in two stages: the first stage uses classifier-free guidance (CFG), and the second stage removes CFG to stabilize training. Since Re-MeanFlow is an efficient one-step model with cheap evaluation, we compute FID on the fly during training (we also provide the corresponding FID statistics).
```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port 9527 train.py \
        --model edm2-img512-s-autog-fid \
        --exp_name remeanflow512\
        --exp_group REMF_512 \
        --results_dir R/512 \
        --t_emb_ckpt R/512/time_emb_512/checkpoints/emb_t_0010000.pt \
        --r_emb_ckpt R/512/time_emb_512/checkpoints/emb_r_0010000.pt \
        --fid_stats_path tools/fid_stats/imagenet512_stats.npz \
        --t_r_schedule re2_truncate \
        --p_std 1.0 \
        --p_mean -0.8 \
        --jvp_api autograd \
        --p 0.5 \
        --pc 1e-2 \
        --w_pi 2.5 \
        --iterations 100000 \
        --stage1_iterations 50000 \
        --ckpt_every 2500 \
        --eval_every 2500 \
        --plot_every 500 \
        --log_every 100 \
        --global_batch_size 128\
        --lr 2e-4 \
        --ema_rate 0.9999 \
        --distance_clip 0.87 \
        #--wandb \
```

## Acknowledgement
This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), and [edm2](https://github.com/NVlabs/edm2) repositories.\
We thank [Haizhou Shi](https://haizhou-shi.github.io/) for many discussions that helped refining this work. We also thank [Charles Hedrick](https://www.cs.rutgers.edu/people/staff/details/charles-hedrick), [Hanz Makmur](https://www.cs.rutgers.edu/people/staff/details/hanz-makmur), and [Timothy Hayes](https://www.cs.rutgers.edu/people/staff/details/timothy-hayes) supporting our computing servers. 


<!-- ## BibTeX
```bibtex
@
}
``` -->