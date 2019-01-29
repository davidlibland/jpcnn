from dataclasses import dataclass


@dataclass
class JPCNNConfig:
    image_dim: int
    num_layers: int=1
    num_resnet: int=1
    num_filters: int=6
    lr: float=1e-3
    epochs: int=500
    description: str="mnist"
    ckpt_interval: int=1
    dropbox_root: str="checkpoints"