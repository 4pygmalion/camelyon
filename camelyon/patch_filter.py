from typing import List

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.errors import GEOSException


class PatchFilter:
    """패치 필터

    Example:
        >>> patchfilter = PatchFilter()
        >>> patchfilter.set_by_optical_density()

        # 필터조건에 하나라도 걸린다면, True
        >>> patchfilter(patch)
        True
    """

    def __init__(self) -> None:
        self.queue = list()

    def __call__(self, patch) -> bool:
        """필터조건에 하나라도 걸린다면 True"""
        for filter_func in self.queue:
            if filter_func(patch):
                return True

        return False

    def add_by_optical_density(self, threshold=0.2) -> None:
        """평균 OD가 threshold보다 낮으면 True"""

        def od_filter(image_array, threshold=threshold) -> bool:
            od_image = -np.log10((image_array + 1e-5) / 255.0)
            return od_image.mean() < threshold

        self.queue.append(od_filter)

        return

    def add_hvs_foregorund_ratio(self, threshold=0.25) -> None:
        """HVS 전경비율이 threshold보다 낮으면 True"""

        def forgound_condition(image_array, threshold: float = threshold) -> bool:
            hsv_image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

            hue = hsv_image_array[..., 0]
            saturation = hsv_image_array[..., 1]
            value = hsv_image_array[..., 2]
            hue_cond = (hue >= 90) & (hue <= 180)
            sat_cond = (saturation >= 8) & (saturation <= 255)
            val_cond = (value >= 103) & (value <= 255)

            return (hue_cond & sat_cond & val_cond).mean() < threshold

        self.queue.append(forgound_condition)

        return
