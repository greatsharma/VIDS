import os
import re
from typing import Callable

from utils import nonmax_suppression


class BaseDetector(object):
    def __init__(
        self,
        initial_frame,
        lane_detector: Callable,
        detection_thresh: float,
    ) -> None:

        self.frame_h, self.frame_w = initial_frame.shape[:2]
        self.lane_detector = lane_detector
        self.detection_thresh = detection_thresh

        self.class_names = [
            "pedestrian",
            "cattles",
            "tw",
            "auto",
            "lmv",
            "hmv",
        ]
        self.num_classes = len(self.class_names)
        self.path_to_yolostuffs = "yolo_stuff/"

        self.yolo_width = 416
        self.yolo_height = 416

        if self.__class__.__name__ == "TrtYoloDetector":
            self._warmup_trt()
        else:
            self._warmup_yolo()

    def _warmup_yolo(self):
        import darknet

        self.config_path = self.path_to_yolostuffs + "yolov4-tiny_416-3.cfg"
        if not os.path.exists(self.config_path):
            raise ValueError(
                "Invalid config path `" + os.path.abspath(self.config_path) + "`"
            )

        weight_path = self.pathconfig_path = self.path_to_yolostuffs + "yolov4-tiny_416-3_final.weights"
        if not os.path.exists(weight_path):
            raise ValueError(
                "Invalid weight path `" + os.path.abspath(weight_path) + "`"
            )

        meta_path = self.pathconfig_path = self.path_to_yolostuffs + "obj.data"
        if not os.path.exists(meta_path):
            raise ValueError(
                "Invalid data file path `" + os.path.abspath(meta_path) + "`"
            )

        self.net_main = darknet.load_net_custom(
            self.config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1
        )

        self.meta_main = darknet.load_meta(meta_path.encode("ascii"))

        try:
            with open(self.meta_main) as metaFH:
                meta_contents = metaFH.read()

                match = re.search(
                    "names *= *(.*)$", meta_contents, re.IGNORECASE | re.MULTILINE
                )

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

        self.darknet_image = darknet.make_image(
            darknet.network_width(self.net_main),
            darknet.network_height(self.net_main),
            3,
        )

    def detect(self, curr_frame) -> list:
        raise NotImplementedError(
            f"detect function of {self.__class__.__name__} is not implemented"
        )

    def _postpreprocessing(self, detections):
        detection_list = []
        ped_and_cattles = []

        for obj_class, obj_prob, obj_bbox in detections:

            if self.__class__.__name__ == "TrtYoloDetector":
                x1 = obj_bbox[0]
                y1 = obj_bbox[1]
                x2 = obj_bbox[2]
                y2 = obj_bbox[3]

            else:
                obj_class = str(obj_class.decode())

                x = obj_bbox[0] * self.frame_w / self.yolo_width
                y = obj_bbox[1] * self.frame_h / self.yolo_height
                w = obj_bbox[2] * self.frame_w / self.yolo_width
                h = obj_bbox[3] * self.frame_h / self.yolo_height

                x1 = x - (w / 2)
                x2 = x + (w / 2)
                y1 = y - (h / 2)
                y2 = y + (h / 2)

            obj_rect = (x1, y1, x2, y2)

            if obj_class in ["pedestrian", "cattles"]:
                ped_and_cattles.append(obj_rect)
                continue

            obj_bottom = (obj_rect[0] + obj_rect[2])/2, obj_rect[3]
            lane = self.lane_detector(obj_bottom)

            if lane is None:
                continue

            detection_list.append(
                {
                    "rect": obj_rect,
                    "obj_bottom": obj_bottom,
                    "obj_class": [obj_class, round(obj_prob, 4)],
                    "lane": lane,
                }
            )

        detection_list = nonmax_suppression(detection_list, 0.55)

        return detection_list, ped_and_cattles
