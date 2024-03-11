"""Tiling

$ python tiling.py \
    -i /home/heon/heon_vast/public_data/Camelyon_Challenge/CAMELYON16 \
    -o ./data \
    -n 48
"""

import os
import glob
import argparse
import multiprocessing
from functools import partial

import tqdm
from core.data_model import WholeSlideImage, CamelyonWSIs, Centers, Labels
from core.patch_filter import PatchFilter


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i", "--input_dir", help="Camleyon16 datadir", required=True
    )
    argparser.add_argument(
        "-o", "--output_dir", help="Output directory to save", required=True
    )
    argparser.add_argument(
        "-n", "--n_jobs", help="Number of jobs", type=int, default=24
    )

    return argparser.parse_args()


def do_tile(whole_slide_image: WholeSlideImage, save_dir: str) -> None:
    """Do tile"""
    patch_filter = PatchFilter()

    if whole_slide_image.label == Labels.BENIGN:
        patch_filter.add_hvs_foregorund_ratio()
        patch_filter.add_by_optical_density()
        whole_slide_image.tile_without_annotation(
            patch_size=512,
            level=1,
            patch_filter=patch_filter,
            save_dir=os.path.join(
                save_dir, Labels.BENIGN.value, whole_slide_image.name
            ),
        )
    else:
        polygons = whole_slide_image.get_polygons()
        patch_filter.add_intersection_over_patch(polygons)
        whole_slide_image.tile_with_annotation(
            polygons,
            label=Labels.MALIGNANT,
            patch_size=512,
            level=1,
            patch_filter=patch_filter,
            save_dir=os.path.join(
                save_dir, Labels.MALIGNANT.value, whole_slide_image.name
            ),
        )

    return


if __name__ == "__main__":
    ARGS: argparse.Namespace = get_args()
    INPUT_DIR = ARGS.input_dir
    OUTPUT_DIR = ARGS.output_dir

    images = list()
    image_paths = glob.glob(os.path.join(INPUT_DIR, "images", "*.tif"))
    for image_path in tqdm.tqdm(image_paths):
        annotation_path = image_path.replace("images", "annotations").replace(
            ".tif", ".xml"
        )
        if not os.path.exists(annotation_path):
            slide_image = WholeSlideImage(slide_path=image_path, label=Labels.BENIGN)
        else:
            slide_image = WholeSlideImage(
                slide_path=image_path,
                annotation_path=annotation_path,
                label=Labels.MALIGNANT,
            )

        images.append(slide_image)

    rumc_slide_images = CamelyonWSIs(
        [
            slide_image
            for slide_image in images
            if slide_image.center == Centers.RUMC.value
        ]
    )
    os.makedirs("data/rumc", exist_ok=True)
    rumc_func = partial(do_tile, save_dir=os.path.join(OUTPUT_DIR, "rumc"))
    with multiprocessing.Pool(ARGS.n_jobs) as pool:
        pool.map(rumc_func, rumc_slide_images)

    umcu_slide_images = CamelyonWSIs(
        [
            slide_image
            for slide_image in images
            if slide_image.center == Centers.UMCU.value
        ]
    )
    os.makedirs("data/umcu", exist_ok=True)
    umcu_func = partial(do_tile, save_dir=os.path.join(OUTPUT_DIR, "umcu"))
    with multiprocessing.Pool(ARGS.n_jobs) as pool:
        pool.map(umcu_func, umcu_slide_images)
