import torch.nn as nn

from codebase.model import ComplexLayers, model_utils


class ComplexDecoder(nn.Module):
    def __init__(self, opt, hidden_dim):
        super().__init__()

        self.opt = opt

        self.out_channel = [
            2 * self.opt.model.hidden_dim,
            2 * self.opt.model.hidden_dim,
            self.opt.model.hidden_dim,
            self.opt.model.hidden_dim,
            self.opt.input.channel,
        ]

        self.conv_model = nn.ModuleList(
            [
                ComplexLayers.ComplexConvTranspose2d(
                    opt,
                    2 * self.opt.model.hidden_dim,
                    self.out_channel[0],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                ),  # e.g. 4x4 => 8x8.
                ComplexLayers.ComplexConv2d(
                    opt,
                    self.out_channel[0],
                    self.out_channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                ComplexLayers.ComplexConvTranspose2d(
                    opt,
                    self.out_channel[1],
                    self.out_channel[2],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                ),  # e.g. 8x8 => 16x16.
                ComplexLayers.ComplexConv2d(
                    opt,
                    self.out_channel[2],
                    self.out_channel[3],
                    kernel_size=3,
                    padding=1,
                ),
                ComplexLayers.ComplexConvTranspose2d(
                    opt,
                    self.out_channel[3],
                    self.out_channel[4],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                ),  # e.g. 16x16 => 32x32.
            ]
        )

        self.hidden_dim = hidden_dim

        linear_out = (
            2 * self.hidden_dim[0] * self.hidden_dim[1] * self.opt.model.hidden_dim
        )
        self.linear = ComplexLayers.ComplexLinear(
            opt, self.opt.model.linear_dim, linear_out
        )

        self.channel_norm = model_utils.init_channel_norm_2d(
            self.out_channel, linear_out, self.opt
        )
