import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class YoloDetector(BaseDetector):

    def __init__(self, is_valid_cntrarea, sub_type: str) -> None:
        super().__init__(is_valid_cntrarea, sub_type)

    def detect(self, curr_frame: np.ndarray) -> list:
        pass
