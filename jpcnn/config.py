from dataclasses import dataclass


@dataclass
class JPCNNConfig:
    image_dim: int
    num_layers: int=2
    num_resnet: int=2
    num_filters: int=64
    lr: float=1e-2
    epochs: int=5
    description: str="tmp"
    ckpt_interval: int=1
    dropbox_root: str="checkpoints"