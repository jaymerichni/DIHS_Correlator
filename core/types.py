from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    transform_type: int
    model_type: str
    random_state: Optional[int] = None
    unknown_class: int = 0


@dataclass
class OutputConfig:
    base_output_dir: str = "./Results"
    save_trees: bool = True
    save_cluster_data: bool = False
    save_untransformed: bool = False

