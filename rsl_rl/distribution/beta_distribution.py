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

        # For numerical stability
        alpha += 1.0e-6
        beta += 1.0e-6

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
        scaled_samples = 2 * samples - 1  # Scale to range [-1, 1]
        log_prob = self.distribution.log_prob(samples).sum(dim=-1)
        return scaled_samples, log_prob

    def log_prob(self, samples):
        unscaled_samples = (samples + 1) / 2  # Rescale back to [0, 1]
        return self.distribution.log_prob(unscaled_samples).sum(dim=-1)

    def entropy(self):
        return self.distribution.entropy()

    def log_info(self):
        return {"sum": (self.alpha + self.beta).mean().item()}
