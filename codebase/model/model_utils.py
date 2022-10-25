import math
import os

import torch
import torch.nn as nn

from codebase.model import ComplexAutoEncoder


def get_model_and_optimizer(opt):
    model = ComplexAutoEncoder.ComplexAutoEncoder(opt)

    print(model)
    print()

    if opt.use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.training.learning_rate)

    return model, optimizer


def init_channel_norm_2d(out_channel, linear_out, opt):
    channel_norm = nn.ModuleList([None] * (len(out_channel) + 1))
    for idx, out_c in enumerate(out_channel):
        channel_norm[idx] = nn.BatchNorm2d(out_c, affine=True)

    channel_norm[-1] = nn.LayerNorm(linear_out, elementwise_affine=True)
    return channel_norm


def save_model(opt, model, optimizer):
    file_path = os.path.join(opt.log_dir, "checkpoint.pt")
    print(f"Saving model to {file_path}.")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, file_path,
    )


def get_conv_biases(out_channels, fan_in):
    magnitude_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
    magnitude_bias = init_magnitude_bias(fan_in, magnitude_bias)

    phase_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
    phase_bias = init_phase_bias(phase_bias)
    return magnitude_bias, phase_bias


def init_phase_bias(bias):
    return nn.init.constant_(bias, val=0)


def init_magnitude_bias(fan_in, bias):
    bound = 1 / math.sqrt(fan_in)
    torch.nn.init.uniform_(bias, -bound, bound)
    return bias
