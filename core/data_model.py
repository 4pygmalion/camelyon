from __future__ import annotations

import os
import numbers
from enum import Enum
from typing import List, Union, Iterable
from dataclasses import dataclass

import tqdm
import openslide
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from matplotlib import pyplot as plt


class Centers(Enum):
    """Centers

    Reference:
        https://academic.oup.com/gigascience/article/7/6/giy065/5026175

    """

    RUMC = "RUMC"
    UMCU = "UMCU"


class Labels(Enum):
    BENIGN = "benign"
    MALIGNANT = "malignant"
    UNKNOWN = "unknown"


@dataclass
class Patch:
    """패치의 데이터클레스

    Args:
        path (str): 패치의 경로
        label (Labels): 패치의 라벨
        name (str): 패치의 이름 (default: "unknown")

    Example:
        // 파일로부터 패치생성
        >>> patch = Patch(path="./image.png", label=Labels.BENIGN, name="image1")
        >>> patch.image_array  # (N, N, 3)

        // 이미지 어레이러부터 패치생성
        >>> image_array = np.array(Image.open("./image.png"))
        >>> patch = Patch.from_array(array, label=Labels.BENIGN)

        // 주피터 노트북에 이미지 렌더링
        >>> patch.plot()

    """

    path: str = None
    label: Labels = None

    name: str = "unknown"
    level: int = None
    x_min: int = None
    y_min: int = None
    x_max: int = None
    y_max: int = None
    center: str = None

    def __post_init__(self):
        if self.path is not None:
            self.name = os.path.basename(self.path)

        return

    def _read_image(self) -> np.ndarray:
        return np.array(Image.open(self.path))

    @property
    def image_array(self):
        if not hasattr(self, "_image_array"):
            return self._read_image()

        return self._image_array

    @image_array.setter
    def image_array(self, image_array):
        self._image_array = image_array
        return

    @classmethod
    def from_file(cls, path: str, label: Labels = None, **kwargs) -> Patch:
        """이미지 어레이러부터 패치생성

        Args:
            array (np.ndarray): 이미지 어레이 (N, N, 3)
            label (Labels): 라벨

        Returns:
            Patch: 패치

        Example:
            >>> image_array = np.array(Image.open("./image.png"))
            >>> patch = Patch.from_array(array, label=Labels.BENIGN)
        """

        patch = cls(label=label, **kwargs)
        patch.image_array = np.array(Image.open(path))

        return patch

    @classmethod
    def from_image_array(cls, image_array: np.ndarray, **kwargs) -> Patch:
        """이미지 어레이러부터 패치생성

        Args:
            array (np.ndarray): 이미지 어레이 (N, N, 3)
            label (Labels): 라벨

        Returns:
            Patch: 패치

        Example:
            >>> image_array = np.array(Image.open("./image.png"))
            >>> patch = Patch.from_array(array, label=Labels.BENIGN)
        """
        patch = cls(**kwargs)
        patch.image_array = image_array

        return patch

    @property
    def polygon(self) -> Union[Polygon, None]:
        if any([xy is None for xy in (self.x_min, self.x_max, self.y_min, self.y_max)]):
            return None

        return Polygon(
            [
                (self.x_min, self.y_min),
                (self.x_max, self.y_min),
                (self.x_max, self.y_max),
                (self.x_min, self.y_max),
            ]
        )

    def __repr__(self) -> str:
        x_min_ymin = str((self.x_min, self.y_min))
        x_max_ymax = str((self.x_max, self.y_max))

        return (
            f"Patch(name={self.name}, label={self.label}, region={x_min_ymin, x_max_ymax}, "
            f"level={self.level}), center={self.center}"
        )

    def plot(self, figsize: tuple = (5, 5)):
        fig, axes = plt.subplots(figsize=figsize)
        axes.imshow(
            self.image_array,
            origin="upper",
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
        )

        return fig, axes

    def plot_over_annotation(self, annotation: Polygon, figsize: tuple = (10, 10)):
        xy = np.array(list(annotation.exterior.coords))
        annotation_x_min, annotaiton_y_min = xy.min(axis=0)
        annotation_x_max, annotation_y_max = xy.max(axis=0)

        image_size = (
            int(annotation_x_max - annotation_x_min),
            int(annotation_y_max - annotaiton_y_min),
        )
        image = Image.new("RGBA", image_size, color="white")
        foreground = Image.fromarray(self.image_array)
        relative_x = int(self.x_min - annotation_x_min)
        relative_y = int(self.y_min - annotaiton_y_min)
        image.paste(
            foreground.resize((self.x_max - self.x_min, self.y_max - self.y_min)),
            (relative_x, relative_y),
        )

        # Annotation
        draw = ImageDraw.Draw(image)
        draw.polygon(
            xy=[
                (int(x - annotation_x_min), int(y - annotaiton_y_min))
                for (x, y) in list(annotation.exterior.coords)
            ],
            outline="red",
            width=int((image_size[0] + image_size[1]) / 2 * 0.002),
        )

        fig, axes = plt.subplots(figsize=figsize)
        axes.imshow(
            image,
            origin="upper",
            extent=[
                annotation_x_min,
                annotation_x_max,
                annotation_y_max,
                annotaiton_y_min,
            ],
        )

        return fig, axes

    def save(self, path: str) -> None:
        """패치 이미지를 주어진 경로에 저장합니다."""
        Image.fromarray(self.image_array).save(path)
        return

    def close(self):
        if not hasattr(self, "_image_array"):
            return

        del self._image_array

        return


@dataclass
class Patches:
    """패치의 복수의 집합

    Example:

        >>> patches = Patches.from_directory(
                "/vast/AI_team/dataset/stomach/leica/train/D",
                Labels.BENIGN
            )
        >>> patch = patches[0]
        >>> patch.plot()

    """

    data: List[Patch]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        n_total = len(self.data)
        n_malignant = self.ys.sum()
        n_benign = n_total - n_malignant

        return f"Patches(len={len(self)}, malignant({n_malignant}), benign({n_benign}))"

    def __add__(self, other: Patches) -> Patches:
        """두 패치들(Patches)을 합친 결과의 Patches을 반환합니다.

        Args:
            other (Patches): 다른 패치집합

        Returns:
            Patches: _description_

        Example:
            >>> train_3dh = Patches.load_patches(ARGS.dh3_dir, "M", label=Labels.MALIGNANT) \+
                    Patches.load_patches(ARGS.dh3_dir, "B", label=Labels.BENIGN)
        """
        if not isinstance(other, Patches):
            raise TypeError(f"other must be Patches, not {type(other)}")

        return Patches(self.data + other.data)

    def __getitem__(self, index) -> Patch:
        if isinstance(index, Iterable):
            if all([isinstance(i, numbers.Integral) for i in index]):
                return Patches([self.data[i] for i in index])

        elif isinstance(index, slice):
            return Patches(self.data[index])

        elif isinstance(index, int):
            return self.data[index]

        return self.data[index]

    @property
    def xs(self) -> np.ndarray:
        """이미지 어레이를 반환함

        Example:
            >>> patches[:10].xs.shape
            (10, 512, 512, 3)

            >>> patches[[0]].xs.shape
            (1, 512, 512, 3)
        """
        if len(self.data) == 0:
            return np.array([])

        return np.stack([patch.image_array for patch in self.data], axis=0)

    @property
    def ys(self) -> np.ndarray:
        if len(self.data) == 0:
            return np.array([])

        labels = list()
        for patch in self.data:
            if patch.label.value == Labels.BENIGN.value:
                labels.append(0)
            elif patch.label.value == Labels.MALIGNANT.value:
                labels.append(1)
            else:
                raise NotImplementedError

        return np.array([labels]).reshape(-1, 1)

    @classmethod
    def from_directory(cls, dir: str, label: Labels) -> Patches:
        """디렉토리로부터 패치집합(Patches)을 생성함

        Args:
            dir (str): 이미지 파일이 있는 경로
            label (Labels): 라벨

        Raises:
            FileNotFoundError: 디렉토리가 없는 경우

        Returns:
            Patches: 패치집합

        Example:
            >>> Patches.from_directory(dir="/vast/AI...", label=Labels.BENIGN)
        """
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Passed {dir} not found")

        data = list()
        for image_path in os.listdir(dir):
            if not image_path.endswith(".png") and not image_path.endswith(".jpg"):
                continue

            patch = Patch(path=os.path.join(dir, image_path), label=label)
            data.append(patch)

        return Patches(data)

    def load(self) -> None:
        for patch in self.data:
            patch.load()

        return

    def save(self, dir: str) -> None:
        """패치들을 특정 디렉토리에 저장합니다."""
        if not os.path.exists(dir):
            os.makedirs(dir)

        for patch in self.data:
            patch_path = os.path.join(dir, patch.name + ".png")
            patch.save(patch_path)

        return

    def close(self) -> None:
        for patch in self.data:
            patch.close()

        return


@dataclass
class WholeSlideImage:
    slide_path: str
    annotation_path: str = str()
    label: Labels = Labels.UNKNOWN

    def __post_init__(self) -> None:
        self.name = os.path.basename(self.slide_path).split(".")[0]
        self.slide = openslide.OpenSlide(self.slide_path)
        self.scanner = self.slide.properties.get("philips.DICOM_MANUFACTURER", str())
        if self.scanner:
            self.center = (
                Centers.RUMC.value
                if self.scanner.startswith("3D")
                else Centers.UMCU.value
            )
        else:
            self.center = str()

        # TODO: ValueError: ctypes objects containing pointers cannot be pickled 해결필요
        del self.slide

        return

    def __repr__(self) -> str:
        return f"WholeSlideImage(name={self.name}, center={self.center})"

    def get_polygons(self) -> List[Polygon]:
        """xml 파일로부터 polygon 정보를 가져옴

        Note:
            XML파일의 포맷은 아래와 같음
            <?xml version="1.0"?>
            <ASAP_Annotations>
            <Annotations>
                <Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="Tumor" Color="#F4FA58">
                        <Coordinates>
                                <Coordinate Order="0" X="55486.6016" Y="59360.1992" />
                                <Coordinate Order="1" X="55496.3008" Y="59257.6992" />
                                <Coordinate Order="2" X="55518.3008" Y="59162.6016" />
                                <Coordinate Order="3" X="55554.8984" Y="59065" />
                                <Coordinate Order="4" X="55586.6016" Y="58965" />

            디폴트로 0레벨에서의 Annotation임.
        """

        if self.annotation_path is None or not os.path.exists(self.annotation_path):
            return list()

        tree = ET.parse(self.annotation_path)
        root = tree.getroot()

        polygons = list()
        for annot in root.iter("Annotation"):
            for region in annot.iter("Coordinates"):
                points = list()
                for coordinate in region.iter("Coordinate"):
                    x = float(coordinate.get("X"))
                    y = float(coordinate.get("Y"))
                    points.append((x, y))

            polygons.append(Polygon(points))

        return polygons

    def get_thumbnail(self, size: tuple = (2024, 2024)):
        slide = openslide.open_slide(self.slide_path)
        thumnail = slide.get_thumbnail(size)

        polygons = list()
        if self.annotation_path:
            polygons = self.get_polygons()
        if not polygons:
            return thumnail

        w, h = slide.level_dimensions[0]
        thumnail_w, thumnail_h = thumnail.size

        correction_w = thumnail_w / w
        correction_h = thumnail_h / h

        draw = ImageDraw.Draw(thumnail)
        for polygon in self.get_polygons():
            xs, ys = polygon.exterior.xy
            xs = np.array(xs) * correction_w
            ys = np.array(ys) * correction_h
            corrected_polygon = Polygon(list(zip(xs, ys)))

            xy_coordinates = [xy for xy in list(corrected_polygon.exterior.coords)]

            draw.polygon(xy_coordinates, outline="blue", width=int(10240 * 0.001))

        return thumnail

    def tile_with_annotation(
        self,
        polygons: List[Polygon],
        label: Labels,
        patch_size: int,
        level: int,
        patch_filter=None,
        save_dir: str = None,
        verbose: bool = False,
    ):
        """Annotation이 있는 경우에 Annotation의 Polygon에 해당하는 패치영역만 추출

        Args:
            patch_size (tuple): 추출하려는 패치의 크기 (width, height).
            level (int): 슬라이드의 해상도 레벨.
            w_stride (int, optional): 가로 스트라이드 크기. 기본값은 None.
            h_stride (int, optional): 세로 스트라이드 크기. 기본값은 None.
            n_sampling (int, optional): 추출할 패치의 최대 샘플링 개수. 기본값은 -1.
            patch_filter (PatchFilter, optional): 패치 필터링을 위한 PatchFilter 객체. 기본값은 None.

        Note:
            Annotation이 있는 경우에만 해당하는 Polygon 영역의 패치를 추출합니다.
            슬라이드 당 소요 시간은 1분 미만입니다.

        Example:
            >>> wsi_processor = WSIProcessor()
            >>> patch_size = (256, 256)
            >>> level = 2
            >>> extracted_patches = wsi_processor.tile_with_annotation(
                    patch_size=patch_size,
                    level=level,
                    w_stride=50,
                    h_stride=50,
                    patch_filter=my_custom_patch_filter
                )
        """
        slide = openslide.open_slide(self.slide_path)

        patch_container = list()
        for polygon in polygons:
            xs, ys = polygon.exterior.xy
            polygon_min_x, polygon_min_y = int(min(xs)), int(min(ys))
            polygon_max_x, polygon_max_y = int(max(xs)), int(max(ys))

            interval = patch_size * (level * 2)
            if verbose:
                y_range = range(polygon_min_y, polygon_max_y, interval)
            else:
                y_range = tqdm.tqdm(range(polygon_min_y, polygon_max_y, interval))

            for patch_min_y in y_range:
                for patch_min_x in range(polygon_min_x, polygon_max_x, interval):
                    patch: Image.Image = slide.read_region(
                        (patch_min_x, patch_min_y), level, (patch_size, patch_size)
                    )

                    patch_max_x = patch_min_x + interval
                    patch_max_y = patch_min_y + interval
                    patch = Patch.from_image_array(
                        image_array=np.array(patch.convert("RGB")),
                        label=label,
                        level=level,
                        x_min=patch_min_x,
                        x_max=patch_max_x,
                        y_min=patch_min_y,
                        y_max=patch_max_y,
                        center=self.center,
                    )

                    if patch_filter is not None and patch_filter(patch):
                        continue

                    patch.name = f"{self.name}_{patch_min_x}_{patch_min_y}_{patch_max_x}_{patch_max_y}"

                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        patch.save(os.path.join(save_dir, patch.name + ".jpg"))
                        continue

                    patch_container.append(patch)

        slide.close()

        return Patches(patch_container)

    def tile_without_annotation(
        self,
        patch_size: int,
        level: int,
        patch_filter=None,
        save_dir: str = None,
    ) -> Patches:
        """Annotation이 없는 경우에 슬라이드의 영역의 패치를 추출

        Args:
            patch_size (tuple): 추출하려는 패치의 크기 (width, height).
            level (int): 슬라이드의 해상도 레벨.
            w_stride (int, optional): 가로 스트라이드 크기. 기본값은 None.
            h_stride (int, optional): 세로 스트라이드 크기. 기본값은 None.
            patch_filter (PatchFilter, optional): 패치 필터링을 위한 PatchFilter 객체. 기본값은 None.

        Note:
            Annotation이 있는 경우에만 해당하는 Polygon 영역의 패치를 추출합니다.
            슬라이드 당 소요 시간은 1분 미만입니다.

        Example:
            >>> wsi_processor = WSIProcessor()
            >>> patch_size = (256, 256)
            >>> level = 2
            >>> extracted_patches = wsi_processor.tile_without_annotation(
                    patch_size=patch_size,
                    level=level,
                    w_stride=50,
                    h_stride=50,
                    n_sampling=100,
                    patch_filter=my_custom_patch_filter
                )
        """

        slide = openslide.open_slide(self.slide_path)
        slide_width, slide_height = slide.dimensions  # for level 0

        patch_container = list()
        interval = patch_size * (level * 2)
        for patch_min_y in tqdm.tqdm(range(0, slide_height, interval)):
            for patch_min_x in range(0, slide_width, interval):
                patch: Image.Image = slide.read_region(
                    (patch_min_x, patch_min_y), level, (patch_size, patch_size)
                )
                patch_max_x = patch_min_x + interval
                patch_max_y = patch_min_y + interval
                patch = Patch.from_image_array(
                    image_array=np.array(patch.convert("RGB")),
                    level=level,
                    x_min=patch_min_x,
                    x_max=patch_max_x,
                    y_min=patch_min_y,
                    y_max=patch_max_y,
                    center=self.center,
                )

                if patch_filter is not None and patch_filter(patch):
                    continue

                patch.name = f"{self.name}_{patch_min_x}_{patch_min_y}_{patch_max_x}_{patch_max_y}"
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    patch.save(os.path.join(save_dir, patch.name + ".jpg"))
                    continue

                patch_container.append(patch)

        slide.close()

        return Patches(patch_container)


@dataclass
class CamelyonWSIs:
    """카멜레온 데이터셋내의 WSI의 집합

    Example:
        데이터 만들기
        >>> malginant_wsis = list()
        >>> for wsi_path in os.listdir(image_dir):
        >>>     image_path = os.path.join(image_dir, wsi_path)
        >>>     annotation_path = os.path.join(annotation_dir, f"{wsi_name}.xml")
        >>>     malginant_wsis.append(
        >>>         CamelyonWSI(
        >>>             image_path, annotation_path, label=Labels.MALIGNANT, name=wsi_name
        >>>         )
        >>>     )
        >>> whole_slide_images = CamelyonWSIs(malginant_wsis)

        # Iteration
        >>> for whole_slide_image in whole_slide_images:
                ...do something...

    """

    data: List[WholeSlideImage]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self) -> str:
        return f"CamelyonWSIs(len={len(self)})"

    def __getitem__(self, index) -> Patch:
        if isinstance(index, Iterable):
            if all([isinstance(i, numbers.Integral) for i in index]):
                return CamelyonWSIs([self.data[i] for i in index])

        elif isinstance(index, slice):
            return CamelyonWSIs(self.data[index])

        elif isinstance(index, int):
            return self.data[index]

        return self.data[index]
