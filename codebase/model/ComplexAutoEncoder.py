import time

import torch
import torch.nn as nn
from einops import rearrange

from codebase.model import ComplexDecoder, ComplexEncoder
from codebase.utils import utils, eval_utils


class ComplexAutoEncoder(nn.Module):
    def __init__(self, opt):
        super(ComplexAutoEncoder, self).__init__()

        self.opt = opt

        self.encoder = ComplexEncoder.ComplexEncoder(opt)
        self.decoder = ComplexDecoder.ComplexDecoder(opt, self.encoder.hidden_dim)

        self.output_model = nn.Conv2d(
            self.opt.input.channel, self.opt.input.channel, 1, 1
        )
        self._init_output_model()

    def _init_output_model(self):
        nn.init.constant_(self.output_model.weight, 1)
        nn.init.constant_(self.output_model.bias, 0)

    def _prepare_input(self, input_images):
        phase = torch.zeros_like(input_images)
        return utils.get_complex_number(input_images, phase)

    def _run_evaluation(self, outputs, labels):
        outputs = eval_utils.apply_kmeans(self.opt, outputs, labels)

        outputs["ARI+BG"] = eval_utils.calc_ari_score(
            self.opt, labels, outputs["labels_pred"], with_background=True
        )
        outputs["ARI-BG"] = eval_utils.calc_ari_score(
            self.opt, labels, outputs["labels_pred"], with_background=False
        )

        return outputs

    def _log_outputs(self, complex_output, reconstruction, outputs):
        outputs["reconstruction"] = reconstruction
        outputs["phase"] = complex_output.angle()
        outputs["norm_magnitude"] = utils.clip_and_rescale(
            complex_output.abs(), self.opt.evaluation.phase_mask_threshold
        )
        return outputs

    def _apply_module(self, module, channel_norm, z):
        m, phi = module(z)
        z = self._apply_activation_function(m, phi, channel_norm)
        return z

    def _apply_activation_function(self, m, phi, channel_norm):
        m = channel_norm(m)
        m = torch.nn.functional.relu(m)
        return utils.get_complex_number(m, phi)

    def _apply_conv_layers(self, model, z):
        for idx, _ in enumerate(model.conv_model):
            z = self._apply_module(model.conv_model[idx], model.channel_norm[idx], z)
        return z

    def encode(self, x):
        # Apply convolutional layers.
        z = self._apply_conv_layers(self.encoder, x)

        # Apply linear layer.
        z = rearrange(z, "b c h w -> b (c h w)")
        z = self._apply_module(self.encoder.linear, self.encoder.channel_norm[-1], z)

        return z

    def decode(self, z):
        # Apply linear layer.
        z = self._apply_module(self.decoder.linear, self.decoder.channel_norm[-1], z)

        z = rearrange(
            z,
            "b (c h w) -> b c h w",
            b=self.opt.input.batch_size,
            h=self.decoder.hidden_dim[0],
            w=self.decoder.hidden_dim[1],
        )

        # Apply convolutional layers.
        complex_output = self._apply_conv_layers(self.decoder, z)

        # Handle output.
        output_magnitude = complex_output.abs()
        reconstruction = self.output_model(output_magnitude)
        reconstruction = torch.sigmoid(reconstruction)

        return reconstruction, complex_output

    def forward(
        self, input_images, labels, step, partition="train", evaluate=False,
    ):
        start_time = time.time()
        complex_input = self._prepare_input(input_images)

        z = self.encode(complex_input)
        reconstruction, complex_output = self.decode(z)

        outputs = {"loss": nn.functional.mse_loss(reconstruction, input_images)}

        if step % self.opt.training.print_idx == 0 or evaluate:
            outputs = self._log_outputs(complex_output, reconstruction, outputs)
            outputs = self._run_evaluation(outputs, labels)

            if partition == "train":
                utils.print_results(partition, step, time.time() - start_time, outputs)

        return outputs
