import os
import glob
from dataclasses import dataclass

from .data_model import Labels, Centers


@dataclass
class CenterData:
    center_path: str

    def __post_init__(self):
        setattr(
            self,
            Labels.benign.name,
            glob.glob(os.path.join(self.center_path, Labels.benign.name, "*.png")),
        )
        setattr(
            self,
            Labels.malignant.name,
            glob.glob(os.path.join(self.center_path, Labels.malignant.name, "*.png")),
        )


@dataclass
class ImagePaths:
    data_root: str

    def __post_init__(self):
        for center in Centers:
            setattr(
                self, center.name, CenterData(os.path.join(self.data_root, center.name))
            )
