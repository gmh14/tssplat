import numpy as np
import nvdiffrast.torch as dr
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from dataclasses import dataclass, field
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.typing import *  # NOQA
from utils.config import parse_structured, get_device  # NOQA
from models import get_encoding, get_mlp, get_activation, scale_tensor  # NOQA


def contract_to_unisphere(
    x, bbox, unbounded: bool = False
):
    if unbounded:
        x = scale_tensor(x, bbox, (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        x = scale_tensor(x, bbox, (0, 1))
    return x


class ExplicitMaterial(torch.nn.Module):
    @dataclass
    class Config:
        n_output_dims: int
        material_activation: str
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )


    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None):
        super(ExplicitMaterial, self).__init__()

        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()

        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )

        self.encoding = get_encoding(
            3, self.cfg.pos_encoding_config
        )
        self.feature_network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_output_dims,
            self.cfg.mlp_network_config,
        )

        self.to(self.device)

    def forward(
        self,
        positions,
        **kwargs,
    ) -> Dict[str, Any]:
        positions = contract_to_unisphere(
            positions, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding(positions.view(-1, 3))
        features = self.feature_network(enc).view(
            *positions.shape[:-1], 3
        )
        
        material = get_activation(self.cfg.material_activation)(
            features
        )
        # color = material.clamp(0.0, 1.0)
        color = material

        out = {
            "color": color,
        }

        return out

    def export(self, path: str, folder: str):
        os.makedirs(os.path.join(path, folder), exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, folder, "material.pth"))
        