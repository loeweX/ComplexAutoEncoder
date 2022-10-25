import torch
import torch.nn as nn

from codebase.model import model_utils
from codebase.utils import utils


def apply_layer(real_function, phase_bias, magnitude_bias, x):
    psi = real_function(x.real) + 1j * real_function(x.imag)
    m_psi = psi.abs() + magnitude_bias
    phi_psi = utils.stable_angle(psi) + phase_bias

    chi = real_function(x.abs()) + magnitude_bias
    m = 0.5 * m_psi + 0.5 * chi

    return m, phi_psi


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        opt,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
    ):
        super(ComplexConvTranspose2d, self).__init__()

        self.opt = opt

        self.conv_tran = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=False,
        )

        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        fan_in = out_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = model_utils.get_conv_biases(
            out_channels, fan_in
        )

    def forward(self, x):
        return apply_layer(self.conv_tran, self.phase_bias, self.magnitude_bias, x)


class ComplexConv2d(nn.Module):
    def __init__(
        self, opt, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
    ):
        super(ComplexConv2d, self).__init__()

        self.opt = opt

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False,
        )

        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = model_utils.get_conv_biases(
            out_channels, fan_in
        )

    def forward(self, x):
        return apply_layer(self.conv, self.phase_bias, self.magnitude_bias, x)


class ComplexLinear(nn.Module):
    def __init__(self, opt, in_channels, out_channels):
        super(ComplexLinear, self).__init__()

        self.opt = opt

        self.fc = nn.Linear(in_channels, out_channels, bias=False)

        self.magnitude_bias, self.phase_bias = self._get_biases(
            in_channels, out_channels
        )

    def _get_biases(self, in_channels, out_channels):
        fan_in = in_channels
        magnitude_bias = nn.Parameter(torch.empty((1, out_channels)))
        magnitude_bias = model_utils.init_magnitude_bias(fan_in, magnitude_bias)

        phase_bias = nn.Parameter(torch.empty((1, out_channels)))
        phase_bias = model_utils.init_phase_bias(phase_bias)
        return magnitude_bias, phase_bias

    def forward(self, x):
        return apply_layer(self.fc, self.phase_bias, self.magnitude_bias, x)
