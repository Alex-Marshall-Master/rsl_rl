#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
from torch.distributions import Beta

# https://github.com/hill-a/stable-baselines/issues/112
# https://keisan.casio.com/exec/system/1180573226


class BetaDistribution(nn.Module):
    def __init__(self, dim, cfg):
        super(BetaDistribution, self).__init__()
        self.output_dim = dim
        self.distribution = None
        self.alpha = None
        self.beta = None
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.sigmoid = nn.Sigmoid()
        print("SCALE CHECK", cfg["scale"])

        if isinstance(cfg["scale"], tuple):
            self.scale = nn.Parameter(torch.Tensor(cfg["scale"]))
            print(self.scale)
            self.scale.requires_grad = False
        else:
            self.scale = cfg["scale"]

    def get_beta_parameters(self, logits):
        ratio = self.sigmoid(logits[:, : self.output_dim])  # (0, 1) a/(a+b) (Mean)
        sum_logits = logits[:, self.output_dim:]  # Extract the second half for sum
        sum = (self.soft_plus(sum_logits) + 1) * self.scale  # (1, ~ (a+b))

        alpha = ratio * sum
        beta = sum - alpha

        # Ensure alpha and beta are strictly positive
        alpha = alpha.clamp(min=1.0e-6)
        beta = beta.clamp(min=1.0e-6)

        # Check for NaN or inf values in alpha and beta
        if torch.isnan(alpha).any() or torch.isinf(alpha).any() or torch.isnan(beta).any() or torch.isinf(beta).any():
            raise ValueError("Alpha or Beta parameters contain NaN or inf values.")
    
        return alpha, beta
    
    def forward(self, logits):
        self.alpha, self.beta = self.get_beta_parameters(logits)
        self.distribution = Beta(self.alpha, self.beta)
        return self.distribution
    
    def mean(self, logits):
        return self.sigmoid(logits[:, : self.output_dim])  # Output is between 0 and 1

    def sample(self, logits):
        self.forward(logits)
        samples = self.distribution.sample()
        # Scale samples to the desired ranges
        scaled_samples = torch.empty_like(samples)
        scaled_samples[:, 0] = torch.clamp(samples[:, 0] * 3.0 - 1.0, -0.999999, 2.999999)  # Scale to range [-1, 2] for vx
        scaled_samples[:, 1] = torch.clamp(samples[:, 1] * 1.5 - 0.75, -0.749999, 0.749999)  # Scale to range [-0.75, 0.75] for vy
        scaled_samples[:, 2] = torch.clamp(samples[:, 2] * 2.5 - 1.25, -1.249999, 1.249999)  # Scale to range [-1.25, 1.25] for v_yaw

        # Ensure numerical stability by clamping values
        log_prob = self.log_prob(scaled_samples)
        return scaled_samples, log_prob

    def log_prob(self, scaled_samples):
        # Rescale back to [0, 1]
        unscaled_samples = torch.empty_like(scaled_samples)
        unscaled_samples[:, 0] = (scaled_samples[:, 0] + 1.0) / 3.0  # Rescale vx to [0, 1]
        unscaled_samples[:, 1] = (scaled_samples[:, 1] + 0.75) / 1.5  # Rescale vy to [0, 1]
        unscaled_samples[:, 2] = (scaled_samples[:, 2] + 1.25) / 2.5  # Rescale v_yaw to [0, 1]

        # Ensure numerical stability by clamping values
        unscaled_samples = torch.clamp(unscaled_samples, 1e-6, 1.0 - 1e-6)

        log_prob = self.distribution.log_prob(unscaled_samples).sum(dim=-1)
        return log_prob

    def entropy(self):
        return self.distribution.entropy()

    def log_info(self):
        return {"sum": (self.alpha + self.beta).mean().item()}
