import cv2
import numpy as np

from utils import rects_from_contours
from detectors import BaseDetector


class FrameDiffDetector(BaseDetector):

    def __init__(self, is_valid_cntrarea, sub_type, prev_frame: np.ndarray) -> None:
        super().__init__(is_valid_cntrarea, sub_type)
        self.prev_frame = prev_frame

    def detect(self, curr_frame: np.ndarray) -> list:
        curr_frame = cv2.cvtColor(curr_frame, code=cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(curr_frame, self.prev_frame)

        if self.sub_type == "prevframe_diff":
            self.prev_frame = curr_frame

        _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        frame_diff = cv2.dilate(frame_diff, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(frame_diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return rects_from_contours(contours, self.is_valid_cntrarea)


class BackgroundSubDetector(BaseDetector):

    def __init__(self, is_valid_cntrarea, sub_type: str) -> None:
        super().__init__(is_valid_cntrarea, sub_type)

        if sub_type == "mog":
            self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        elif sub_type == "mog2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    def detect(self, curr_frame: np.ndarray) -> list:
        curr_frame = cv2.cvtColor(curr_frame, code=cv2.COLOR_BGR2GRAY)
        fgmask = self.bg_subtractor.apply(curr_frame)
        contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return rects_from_contours(contours, self.is_valid_cntrarea)
