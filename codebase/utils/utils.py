import os
import pickle
import random
from datetime import timedelta

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from omegaconf import open_dict


def parse_args(opt):
    with open_dict(opt):
        opt.log_dir = os.getcwd()
        print(f"Logging files in {opt.log_dir}")
        opt.device = "cuda:0" if opt.use_cuda else "cpu"
        opt.cwd = get_original_cwd()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    save_opt(opt)
    print(OmegaConf.to_yaml(opt))
    return opt


def save_opt(opt):
    file_path = os.path.join(opt.log_dir, "opt.pkl")
    with open(file_path, "wb") as opt_file:
        pickle.dump(opt, opt_file)


def get_learning_rate(opt, step):
    if opt.training.learning_rate_schedule == 0:
        return opt.training.learning_rate
    elif opt.training.learning_rate_schedule == 1:
        return get_linear_warmup_lr(opt, step)
    else:
        raise NotImplementedError


def get_linear_warmup_lr(opt, step):
    if step < opt.training.warmup_steps:
        return opt.training.learning_rate * step / opt.training.warmup_steps
    else:
        return opt.training.learning_rate


def update_learning_rate(optimizer, opt, step):
    lr = get_learning_rate(opt, step)
    optimizer.param_groups[0]["lr"] = lr
    return optimizer, lr


def tensor_to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def spherical_to_cartesian_coordinates(x):
    # Second dimension of x contains spherical coordinates: (r, phi_1, ... phi_n).
    num_dims = x.shape[1]
    out = torch.zeros_like(x)

    r = x[:, 0]
    phi = x[:, 1:]

    sin_component = 1
    for i in range(num_dims - 1):
        out[:, i] = r * torch.cos(phi[:, i]) * sin_component
        sin_component = sin_component * torch.sin(phi[:, i])

    out[:, -1] = r * sin_component
    return out


def phase_to_cartesian_coordinates(opt, phase, norm_magnitude):
    # Map phases on unit-circle and transform to cartesian coordinates.
    unit_circle_phase = torch.concat(
        (torch.ones_like(phase)[:, None], phase[:, None]), dim=1
    )

    if opt.evaluation.phase_mask_threshold != -1:
        # When magnitude is < phase_mask_threshold, use as multiplier to mask out respective phases from eval.
        unit_circle_phase = unit_circle_phase * norm_magnitude[:, None]

    return spherical_to_cartesian_coordinates(unit_circle_phase)


def clip_and_rescale(input_tensor, clip_value):
    if torch.is_tensor(input_tensor):
        clipped = torch.clamp(input_tensor, min=0, max=clip_value)
    elif isinstance(input_tensor, np.ndarray):
        clipped = np.clip(input_tensor, a_min=0, a_max=clip_value)
    else:
        raise NotImplementedError

    return clipped * (1 / clip_value)


def get_complex_number(magnitude, phase):
    return magnitude * torch.exp(phase * 1j)


def complex_tensor_to_real(complex_tensor, dim=-1):
    return torch.stack([complex_tensor.real, complex_tensor.imag], dim=dim)


def stable_angle(x: torch.tensor, eps=1e-8):
    """ Function to ensure that the gradients of .angle() are well behaved."""
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return y.angle()


def print_results(partition, step, time_spent, outputs):
    print(
        f"{partition} \t \t"
        f"Step: {step} \t"
        f"Time: {timedelta(seconds=time_spent)} \t"
        f"MSE Loss: {outputs['loss'].item():.4e} \t"
        f"ARI+BG: {outputs['ARI+BG']:.4e} \t"
        f"ARI-BG: {outputs['ARI-BG']:.4e} \t"
    )
