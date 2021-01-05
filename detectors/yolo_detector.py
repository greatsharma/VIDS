import os
import re
import cv2
import darknet
import numpy as np

from detectors.base_detector import BaseDetector


class YoloDetector(BaseDetector):

    def __init__(self, initial_frame: np.ndarray, is_valid_cntrarea=None, sub_type=None) -> None:
        super().__init__(is_valid_cntrarea, sub_type)

        self.config_path = "./cfg/yolov4.cfg"
        if not os.path.exists(self.config_path):
            raise ValueError("Invalid config path `" + os.path.abspath(self.config_path)+"`")

        self.weight_path = "./yolov4.weights"
        if not os.path.exists(self.weight_path):
            raise ValueError("Invalid weight path `" + os.path.abspath(self.weight_path)+"`")

        self.meta_path = "./cfg/coco.data"
        if not os.path.exists(self.meta_path):
            raise ValueError("Invalid data file path `" + os.path.abspath(self.meta_path)+"`")

        self.net_main = darknet.load_net_custom(self.config_path.encode("ascii"), self.weight_path.encode("ascii"), 0, 1)  # batch size = 1

        self.meta_main = darknet.load_meta(self.meta_main.encode("ascii"))

        try:
            with open(self.meta_main) as metaFH:
                meta_contents = metaFH.read()

                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            names_list = namesFH.read().strip().split("\n")
                            self.alt_names = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass

        # Create an image we reuse for each detect
        frame_height, frame_width = initial_frame.shape
        self.darknet_image = darknet.make_image(frame_width, frame_height, 3)  # Create image according darknet for compatibility of network

    def detect(self, curr_frame: np.ndarray) -> list:
        curr_frame = cv2.cvtColor(curr_frame, code=cv2.COLOR_BGR2RGB)

        darknet.copy_image_from_bytes(self.darknet_image, curr_frame.tobytes())

        return darknet.detect_image(self.net_main, self.meta_main, self.darknet_image, thresh=0.3)
