import torch
import numpy as np
from functools import partial
import scipy.stats as stats


###### Ushape t-distribution from 2-rectified flow++
def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)

def sample_t_re2(exponential_pdf, num_samples, a):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def forward_with_cfg(x, t, y, cfg_scale, model):
    t = 1.0 - t
    uncond_cls = torch.ones_like(y) * 1000
    
    model_pred = model(x, t, t, y)
    model_pred_uncond = model(x, t, t, uncond_cls)

    cfg_model_pred = cfg_scale * (model_pred) + (1 - cfg_scale) * (model_pred_uncond)

    return -cfg_model_pred

def build_edm_labels(c: torch.Tensor, num_classes: int = 1000):
    device = c.device

    labels = torch.eye(num_classes, device=device)
    zero = torch.zeros((1, num_classes), device=device)
    labels = torch.cat([labels, zero], dim=0)[c].contiguous()

    return labels

class MeanFlow:
    def __init__(self,
                 model,
                 encoder=None,
                 # data parameters
                 image_size = 32,
                 in_channel = 3,
                 # time parameters
                 t_r_schedule = "re2_truncate",
                 p_std = 1.0, 
                 p_mean = -0.4,
                 # loss parameters
                 consistency_ratio = 0.25, 
                 jvp_api = "autograd",
                 p = 2.0,
                 pc = 1e-3,
                 loss_type = "adaptive_loss_weight",
                 class_dropout = 0.1,
                 # cfg parameters
                 w = 1.0,
                 w_pi = 1.0
                 ):

        self.image_size = image_size
        self.in_channel = in_channel
        self.t_r_schedule = t_r_schedule
        self.exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')
        self.p_std = p_std
        self.p_mean = p_mean
        self.consistency_ratio = consistency_ratio
        self.p = p
        self.pc = pc
        self.loss_type = loss_type
        self.model = model
        self.jvp_api = jvp_api

        # edm2
        self.min_sigma = 1e-8
        self.max_sigma = 80/81

        # cfg parameters
        self.w = w
        self.w_pi = w_pi
        self.class_dropout = class_dropout

        # initialize jvp function   
        if self.jvp_api == "autograd":
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        else:
            raise ValueError(f"JVP API {self.jvp_api} not supported")
        
    def get_xt(self, x, n, t):
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x + t * n
    
    def mean_flatten(self, x):
        return x.mean(dim=(1, 2, 3))
    
    def sample_logit_normal(self, mean, std, shape, device):
        eps = torch.randn(shape, device=device)
        x = eps * std + mean  # X ~ N(mean, std^2)
        return torch.sigmoid(x)  # Y = sigmoid(X) ∈ (0, 1)
        
    def sample_t_and_r_uniform(self, bs, device):
        t = torch.rand(bs, device=device)
        r = t * torch.rand(bs, device=device)

        mask = torch.rand(bs, device=device) < (1 - self.consistency_ratio)
        r[mask] = t[mask]

        return t, r, mask
    
    def sample_t_and_r_re2_truncate(self, bs, device):
        # sample t from ushaped t-distribution
        a=4
        t = sample_t_re2(self.exponential_distribution, bs, a).numpy()
        t = torch.from_numpy(t).float().to(device)
        t = torch.clamp(t, 0, self.max_sigma)

        # sample interval from logit normal distribution
        interval = self.sample_logit_normal(self.p_mean, self.p_std, (bs,), device)
        r = t - interval
        r = torch.clamp(r, 0, 1-1e-5)
        
        # clamp those r with smaller than 0.4 to 0 to avoid high variance interval
        r = torch.where((t > 0.8) & (r < 0.4), torch.zeros_like(r, device=device), r)

        # consistency ratio
        mask = torch.rand(bs, device=device) < (1 - self.consistency_ratio)
        r[mask] = t[mask]

        return t, r, mask
    
    def get_loss(self, error, t=None):
        if self.loss_type == "adaptive_loss_weight":
            sq_norm_error = torch.mean(error**2, dim=(1, 2, 3), keepdim=False)
            weight = 1.0/(sq_norm_error + self.pc).pow(self.p)
            loss = weight.detach() * sq_norm_error 
            return loss.mean()
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported")

    def loss_edm(self, x, c, noise=None):
        bs = x.shape[0]

        # sample t and r 
        if self.t_r_schedule == "uniform":
            t, r, _ = self.sample_t_and_r_uniform(bs, x.device)
        elif self.t_r_schedule == "re2_truncate":
            t, r, _ = self.sample_t_and_r_re2_truncate(bs, x.device)

        uncond = torch.ones_like(c, device=x.device)*1000
        class_dropout_mask = torch.rand(c.shape[0], device=x.device) < self.class_dropout
        c = torch.where(class_dropout_mask, uncond, c)
        c = build_edm_labels(c)
        uncond = build_edm_labels(uncond)

        n = torch.randn_like(x, device=x.device) if noise is None else noise
        xt = self.get_xt(x, n, t)
        v = n - x

        # apply cfg training when w_pi > 1
        if self.w_pi > 1.0:
            with torch.no_grad():
                w = self.w
                k = 1 - w/self.w_pi

                u_uncond = self.model(xt, t, t, uncond)
                u_cond = self.model(xt, t, t, c)

                v_hat = w * v + k * u_cond + (1 - w - k) * u_uncond
                v_hat = v_hat.detach()
        else:
            v_hat = v.detach()

        # wrap model prediction
        model_wrapper = partial(self.model, y=c)
        
        # compute jvp
        jvp_args = (
            lambda xt, t, r: model_wrapper(xt, t, r),
            (xt, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r))
        )
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        # compute loss
        u_tgt = v_hat - (t - r).view(-1, 1, 1, 1) * dudt
        error = u - u_tgt.detach()

        loss = self.get_loss(error)

        return {"loss": loss}

    #########################################################
    # Sampling Methods
    ########################################################
    # one step sampling
    def sample_one_step(self, model, class_idx=[207, 360, 387, 974, 88, 979, 417, 279], n=None, device="cuda"):

        model.eval()
        if not isinstance(class_idx, torch.Tensor):
            c = torch.tensor(class_idx, device=device)
        else:
            c = class_idx.to(device)
        
        c = build_edm_labels(c)
        if n is None:
            n = torch.randn(len(class_idx), self.in_channel, self.image_size, self.image_size, device=device)
        
        t = torch.ones(c.shape[0], device=device)  
        t = t.clamp(0.0, self.max_sigma)
        
        r = torch.zeros(c.shape[0], device=device)
        x = n - model(n, t, r, c)
        return x
    
    # multi-step sampling
    def sample_edm(self, model, class_idx=[207, 360, 387, 974, 88, 979, 417, 279], step_num=100, n=None, device="cuda", og=False):

        model.eval()
        if not isinstance(class_idx, torch.Tensor):
            c = torch.tensor(class_idx, device=device)
        else:
            c = class_idx.to(device)
        
        c = build_edm_labels(c)
        if n is None:
            n = torch.randn(len(class_idx), self.in_channel, self.image_size, self.image_size, device=device)
        
        t_steps = torch.linspace(self.max_sigma, self.min_sigma, step_num + 1, device=device)
        z = n
        with torch.no_grad():
            for i in range(step_num):
                t = t_steps[i]
                dt = t_steps[i+1] - t

                t_tensor = torch.full((z.size(0),), t, device=device)  # batch-wise t

                velocity = model(z, t_tensor, t_tensor, c) if not og else model.forward_og(z, t_tensor, t_tensor, c)
                z = z + dt * velocity
        
        return z