import os
import re
import cv2
import numpy as np

from detectors import darknet
from detectors import BaseDetector


class YoloDetector(BaseDetector):

    def __init__(self, initial_frame: np.ndarray, yolo_weight, is_valid_cntrarea=None, sub_type=None) -> None:
        super().__init__(is_valid_cntrarea, sub_type)

        if yolo_weight == "coco_pretrained":
            self.objects_of_interests = ["tw", "car", "lgv", "bus", "ml", "auto", "mb", "tractr", "2t", "3t", "4t", "5t", "6t"]
        else:
            self.objects_of_interests = ["bicycle", "car", "motorbike", "bus", "truck"]

        base_path = f"detectors/yolo_weights/{yolo_weight}/"

        config_path = base_path + "yolov4.cfg"
        if not os.path.exists(config_path):
            raise ValueError("Invalid config path `" + os.path.abspath(config_path)+"`")

        weight_path = base_path + "yolov4.weights"
        if not os.path.exists(weight_path):
            raise ValueError("Invalid weight path `" + os.path.abspath(weight_path)+"`")

        meta_path = base_path + "obj.data"
        if not os.path.exists(meta_path):
            raise ValueError("Invalid data file path `" + os.path.abspath(meta_path)+"`")

        self.net_main = darknet.load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1

        self.meta_main = darknet.load_meta(meta_path.encode("ascii"))

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

    def _bbox2rect(self, bbox):
        x, y, w, h = bbox
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return (xmin, ymin, xmax, ymax)

    def detect(self, curr_frame: np.ndarray) -> list:
        curr_frame = cv2.cvtColor(curr_frame, code=cv2.COLOR_BGR2RGB)

        darknet.copy_image_from_bytes(self.darknet_image, curr_frame.tobytes())

        detections = darknet.detect_image(self.net_main, self.meta_main, self.darknet_image, thresh=0.3)

        rects = []
        for obj_class, obj_prob, obj_bbox in detections:
            obj_class = str(obj_class.decode())
            if obj_class in self.objects_of_interests:
                rects.append(self._bbox2rect(obj_bbox))

        return rects
