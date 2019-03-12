from dataclasses import dataclass, field
from typing import List


@dataclass
class JPCNNConfig:
    image_dim: int
    num_layers: int=1
    num_resnet: int=6
    avg_num_filters: int=8
    mixtures_per_channel: int=2
    lr: float=1e-3
    epochs: int=500
    description: str="mnist"
    ckpt_interval: int=1
    dropbox_root: str="checkpoints"
    dropbox_sync: bool=True
    display_images: bool=False
    compression: List[List[float]]=field(default_factory=lambda: [[1]])
    seed: int=1
    num_test_elements: int=4
