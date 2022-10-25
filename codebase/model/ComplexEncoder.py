import torch
import torch.nn as nn

from codebase.model import ComplexLayers, model_utils


class ComplexEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.out_channel = [
            self.opt.model.hidden_dim,
            self.opt.model.hidden_dim,
            2 * self.opt.model.hidden_dim,
            2 * self.opt.model.hidden_dim,
            2 * self.opt.model.hidden_dim,
        ]

        self.conv_model = nn.ModuleList(
            [
                ComplexLayers.ComplexConv2d(
                    opt,
                    self.opt.input.channel,
                    self.out_channel[0],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),  # e.g. 32x32 => 16x16.
                ComplexLayers.ComplexConv2d(
                    opt,
                    self.out_channel[0],
                    self.out_channel[1],
                    kernel_size=3,
                    padding=1,
                ),
                ComplexLayers.ComplexConv2d(
                    opt,
                    self.out_channel[1],
                    self.out_channel[2],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),  # e.g. 16x16 => 8x8.
                ComplexLayers.ComplexConv2d(
                    opt,
                    self.out_channel[2],
                    self.out_channel[3],
                    kernel_size=3,
                    padding=1,
                ),
                ComplexLayers.ComplexConv2d(
                    opt,
                    self.out_channel[3],
                    self.out_channel[4],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),  # e.g. 8x8 => 4x4.
            ]
        )

        self.hidden_dim = self.get_hidden_dimension()
        self.linear = ComplexLayers.ComplexLinear(
            opt,
            2 * self.hidden_dim[0] * self.hidden_dim[1] * self.opt.model.hidden_dim,
            self.opt.model.linear_dim,
        )

        self.channel_norm = model_utils.init_channel_norm_2d(
            self.out_channel, self.opt.model.linear_dim, self.opt
        )

    def get_hidden_dimension(self):
        x = torch.zeros(
            1,
            self.opt.input.channel,
            self.opt.input.image_height,
            self.opt.input.image_width,
        )
        for module in self.conv_model:
            x = module.conv(x)

        return x.shape[2], x.shape[3]
