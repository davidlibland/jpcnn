from dataclasses import dataclass, field
from typing import List


@dataclass
class JPCNNConfig:
    image_dim: int
    num_layers: int=1
    num_resnet: int=1
    num_filters: int=16
    lr: float=1e-3
    epochs: int=500
    description: str="mnist"
    ckpt_interval: int=1
    dropbox_root: str="checkpoints"
    display_images: bool=True
    compression: List[List[float]]=field(default_factory=lambda: [[1]])
