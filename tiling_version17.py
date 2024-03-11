"""Tiling

$ python tiling_version17.py \
    -i /home/heon/heon_vast/public_data/Camelyon_Challenge/CAMELYON17 \
    -o /home/heon/heon_vast/camelyon17_patches \
    -n 24

$ tree -L [output_path]    
[output_path]/
├── CWZ
│   ├── benign
│   └── malignant
├── LPON
│   ├── benign
│   └── malignant
├── RST
│   ├── benign
│   └── malignant
├── RUMC
│   ├── benign
│   └── malignant
└── UMCU
    ├── benign
    └── malignant
"""

import os
import argparse

import tqdm
import pandas as pd
from core.data_model import WholeSlideImage, Centers, Labels
from core.patch_filter import PatchFilter
from utils import get_logger


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i", "--input_dir", help="Camleyon17 datadir", required=True
    )
    argparser.add_argument(
        "-o", "--output_dir", help="Output directory to save", required=True
    )
    argparser.add_argument(
        "-n", "--n_jobs", help="Number of jobs", type=int, default=24
    )

    return argparser.parse_args()


def get_slide2center(stage_csv_path: str) -> dict:
    df = pd.read_csv(stage_csv_path)
    df = df.loc[df["patient"].str.endswith(".tif")][["patient", "center"]]

    return dict(zip(df["patient"], df["center"]))


if __name__ == "__main__":
    ARGS: argparse.Namespace = get_args()
    INPUT_DIR = ARGS.input_dir
    OUTPUT_DIR = ARGS.output_dir
    LOGGER = get_logger("camelyon17_tiling")

    slide2center: dict = get_slide2center(os.path.join(INPUT_DIR, "stages.csv"))

    # mkdir
    for center in Centers:
        for label in [Labels.benign.name, Labels.malignant.name]:
            dir = os.path.join(OUTPUT_DIR, center.name, label)
            LOGGER.info(f"Make dir ({dir})")
            os.makedirs(dir, exist_ok=True)

    patch_filter = PatchFilter()
    patch_filter.add_by_optical_density()
    patch_filter.add_hvs_foregorund_ratio()

    LOGGER.info(f"Tiling start")
    for slide_fname, center_num in tqdm.tqdm(slide2center.items()):
        slide_path = os.path.join(INPUT_DIR, "images", slide_fname)
        annotation_path = slide_path.replace("images", "annotations").replace(
            "tif", "xml"
        )
        if not os.path.exists(annotation_path):
            LOGGER.info(f"slide_fname({slide_fname}) did have annotaion")
            continue

        wsi = WholeSlideImage(
            slide_path, annotation_path, center=Centers(center_num), logger=LOGGER
        )
        output_center_dir = os.path.join(OUTPUT_DIR, Centers(center_num).name)
        LOGGER.info(f"Do tiling({slide_fname}), Center:{Centers(center_num).name}")

        wsi.tile_with_full_res(
            patch_size=512,
            overlap=0,
            patch_filter=patch_filter,
            save_dir=output_center_dir,
            n_worker=ARGS.n_jobs,
        )
