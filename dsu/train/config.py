import json
from dataclasses import dataclass, field
from typing import Dict, List

from dacite import from_dict


@dataclass
class Config:
    use_wandb: bool = True
    wandb_project_name: str = "cifar-10"
    model_name: str = "resnet"
    batch_size: int = 128
    num_runs: int = 5
    epochs: int = 200
    val_split: float = 0.1
    augment: bool = True
    momentum: float = 0.9
    lr_decay_values: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.001])
    lr_boundaries: List[int] = field(default_factory=lambda: [32000, 48000])
    augment_pad_size: int = 4

    @classmethod
    def from_file(cls, cfg_file: str) -> "Config":
        with open(cfg_file) as f:
            return from_dict(data_class=Config, data=json.load(f))

    @classmethod
    def from_dict(cls, cfg: Dict[str, any]) -> "Config":
        return from_dict(data_class=Config, data=cfg)
