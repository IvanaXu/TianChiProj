# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import math
from typing import Tuple

import cv2
import numpy as np


class NPImage:
    def __init__(self, meter_per_pixel, width, height, depth=1, data_type=np.uint8):
        self.meter_per_pixel = meter_per_pixel
        self.width = width
        self.height = height
        self.img_data = np.zeros((width, height, depth), np.uint8)
        self.base_pose = None

    def set_center_pose(self, center_pose):
        self.base_pose = (
            center_pose[0] - self.width * self.meter_per_pixel / 2.0,
            center_pose[1] - self.height * self.meter_per_pixel / 2.0,
        )

    def _to_image_point(self, tgt_pose):
        return [
            (tgt_pose[0] - self.base_pose[0]) / self.meter_per_pixel,
            self.height - (tgt_pose[1] - self.base_pose[1]) / self.meter_per_pixel,
        ]

    def draw_rect(self, pose: Tuple, length: float, width: float, color):
        local_points = [
            self._to_image_point(
                (
                    pose[0] + (math.cos(pose[2]) * length - math.sin(pose[2]) * width) / 2,
                    pose[1] + (math.sin(pose[2]) * length + math.cos(pose[2]) * width) / 2,
                ),
            ),
            self._to_image_point(
                (
                    pose[0] + (math.cos(pose[2]) * -length - math.sin(pose[2]) * width) / 2,
                    pose[1] + (math.sin(pose[2]) * -length + math.cos(pose[2]) * width) / 2,
                )
            ),
            self._to_image_point(
                (
                    pose[0] + (math.cos(pose[2]) * -length - math.sin(pose[2]) * -width) / 2,
                    pose[1] + (math.sin(pose[2]) * -length + math.cos(pose[2]) * -width) / 2,
                )
            ),
            self._to_image_point(
                (
                    pose[0] + (math.cos(pose[2]) * length - math.sin(pose[2]) * -width) / 2,
                    pose[1] + (math.sin(pose[2]) * length + math.cos(pose[2]) * -width) / 2,
                )
            ),
        ]
        local_points = np.array([local_points], dtype=np.int32)
        cv2.fillPoly(self.img_data, local_points, (color))

    def draw_polyline(self, pts, color):
        img_pts = np.array([self._to_image_point(pt) for pt in pts], np.int32)
        img_pts = img_pts.reshape((-1, 1, 2))
        cv2.polylines(self.img_data, [img_pts], False, (color), thickness=2)
        return self

    def resize(self, size):
        self.img_data = cv2.resize(
            self.img_data,
            (size, size),
            interpolation=cv2.INTER_CUBIC,
        )
        return self

    def rotate(
        self,
        center,
        angle,
        size,
        scale=1.0,
    ):
        rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        self.img_data = cv2.warpAffine(self.img_data, rotation_matrix, (size, size))
        return self

    def flip(self):
        self.img_data = cv2.flip(self.img_data, 1)
        return self

    def merge(self, np_imgs):
        self.img_data = cv2.merge([self.img_data] + [np_img.img_data for np_img in np_imgs])
        return self

    def to_split_np(self):
        np_data = [cv2.split(self.img_data)]
        return None

    def img_write(self, path):
        cv2.imwrite(
            path,
            self.img_data,
        )
