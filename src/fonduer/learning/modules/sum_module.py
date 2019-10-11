import torch
import torch.nn as nn


class Sum_module(nn.Module):
    def __init__(self, sum_output_keys):
        super().__init__()

        self.sum_output_keys = sum_output_keys

    def forward(self, intermediate_output_dict):
        return torch.stack(
            [intermediate_output_dict[key][0] for key in self.sum_output_keys], dim=0
        ).sum(dim=0)
