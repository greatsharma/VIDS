import cv2
import time
import argparse
import numpy as np

from camera_metadata import CAMERA_METADATA
from detectors import FrameDiffDetector, BackgroundSubDetector, YoloDetector
from trackers import CentroidTracker, KalmanTracker
from utils import draw_text_with_backgroud, init_adpative_cntrarea, init_lane_detector, init_direction_detector


class VehicleTracking(object):

    def __init__(
            self,
            input_path,
            detector_type,
            tracker_type,
            bg_path,
            max_track_pts,
            path_from_centroid,
            max_absent,
            mode):

        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.bg_path = bg_path
        self.max_track_pts = max_track_pts
        self.path_from_centroid = path_from_centroid
        self.max_absent = max_absent
        self.mode = mode

        camera_id = input_path[input_path.find("place")+5]
        self.camera_meta = CAMERA_METADATA[camera_id]

        self.vidcap = cv2.VideoCapture(input_path)

        self._init_detector()
        self._init_tracker()

    def _init_detector(self):
        initial_frame = None
        if self.detector_type == "staticbg_diff":
            initial_frame = cv2.imread(self.bg_path)
        else:
            _, initial_frame = self.vidcap.read()

        self.frame_h, self.frame_w = tuple(d//2 for d in initial_frame.shape[:2])
        initial_frame = cv2.resize(initial_frame, dsize=(self.frame_w, self.frame_h), interpolation=cv2.INTER_AREA)
        initial_frame = cv2.cvtColor(initial_frame, code=cv2.COLOR_BGR2GRAY)

        is_valid_cntrarea = init_adpative_cntrarea(self.camera_meta)

        if self.detector_type in ["prevframe_diff", "staticbg_diff"]:
            self.detector = FrameDiffDetector(is_valid_cntrarea, self.detector_type, initial_frame)
        elif self.detector_type in ["mog", "mog2", "knn"]:
            self.detector = BackgroundSubDetector(is_valid_cntrarea, self.detector_type)
        else:
            self.detector = YoloDetector(initial_frame)

        self.img_for_text = np.zeros((self.frame_h, self.frame_w//3, 3), dtype=np.uint8)

    def _init_tracker(self):
        lane_detector = init_lane_detector(self.camera_meta)
        direction_detector = init_direction_detector(self.camera_meta)
        maxdist = self.camera_meta["max_distance"]

        if self.tracker_type == "centroid":
            self.tracker = CentroidTracker(
                lane_detector, direction_detector, maxdist, self.path_from_centroid, self.max_absent, self.max_track_pts)
        else:
            self.tracker = KalmanTracker(
                lane_detector, direction_detector, maxdist, self.path_from_centroid, self.max_absent, self.max_track_pts)

    def _draw_tracked_objects(self, frame, tracked_objs):
        for obj in tracked_objs.values():
            obj_rect = obj.rect
            obj_ctr = (
                obj.centroid
                if self.tracker_type == "centroid" else
                obj.state[:2]
            )

            cv2.circle(frame, obj_ctr, radius=3, color=(0, 0, 0), thickness=-1)

            if obj.direction:
                base_color = [0, 255, 0]
                cv2.rectangle(frame, obj_rect[:2], obj_rect[2:], base_color, 2)
                draw_text_with_backgroud(frame, f"{obj.lane} lane", obj_ctr[0]-10, obj_ctr[1]-10, font_scale=0.35, thickness=1)
            else:
                base_color = [0, 0, 255]
                cv2.rectangle(frame, obj_rect[:2], obj_rect[2:], base_color, 2)
                draw_text_with_backgroud(frame, "Wrong Direction", obj_ctr[0]-10, obj_ctr[1]-10, font_scale=0.4, thickness=1, foreground=(0, 0, 255))

            prev_point = None
            for pt, perc, size in zip(obj.path, np.linspace(0.25, 0.75, len(obj.path)), [2]*10 + [3]*15 + [4]*15):
                if not prev_point is None:
                    color = tuple(np.array(base_color)*(1-perc))
                    cv2.line(frame, (prev_point[0], prev_point[1]), (pt[0], pt[1]), color, thickness=size, lineType=8)
                prev_point = pt

    def run(self):
        frame_count = 0
        tik = time.time()

        while self.vidcap.isOpened():
            status, frame = self.vidcap.read()
            if not status:
                break

            frame_count += 1

            frame = cv2.resize(frame, dsize=(self.frame_w, self.frame_h), interpolation=cv2.INTER_AREA)

            detections = self.detector.detect(frame)
            tracked_objects = self.tracker.update(detections)

            self._draw_tracked_objects(frame, tracked_objects)

            fps = frame_count // (time.time() - tik + 1e-6)

            if self.mode == "debug":
                # road tracks
                cv2.polylines(frame, [self.camera_meta["leftlane_coords"]], isClosed=True, color=(255, 0, 0), thickness=2)
                cv2.polylines(frame, [self.camera_meta["rightlane_coords"]], isClosed=True, color=(255, 0, 0), thickness=2)
                # reference points
                cv2.circle(frame, self.camera_meta["leftlane_ref"], radius=2, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, self.camera_meta["rightlane_ref"], radius=2, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, self.camera_meta["mid_ref"], radius=2, color=(0, 0, 255), thickness=-1)

            draw_text_with_backgroud(self.img_for_text, "VIDS", x=15, y=30, font_scale=1, thickness=2)
            draw_text_with_backgroud(self.img_for_text, f"Detector: {self.detector_type}", x=15, y=60, font_scale=0.5, thickness=1)
            draw_text_with_backgroud(self.img_for_text, f"Tracker: {self.tracker_type}", x=15, y=80, font_scale=0.5, thickness=1)
            draw_text_with_backgroud(self.img_for_text, f"FPS: {fps}", x=15, y=100, font_scale=0.5, thickness=1)

            out_frame = np.hstack((frame, self.img_for_text))
            cv2.imshow("VIDS", out_frame)

            key = cv2.waitKey(1)

            if key == ord("p"):
                cv2.waitKey(-1)
            elif key == ord("q"):
                break

        self.vidcap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('-i', '--input', type=str, required=True, help='path to input video')

    ap.add_argument('-d', '--detector', type=str, required=False, default="prevframe_diff",
                    help="detector to use", choices=["prevframe_diff", "staticbg_diff", "mog", "mog2", "knn", "yolo"])

    ap.add_argument('-t', '--tracker', type=str, required=False, default="kalman",
                    help="tracker to use", choices=["centroid", "kalman"])

    ap.add_argument('-bg', '--background', type=str, required=False,
                    help='path to background frame, needed only for `staticbg_diff` detector')

    ap.add_argument('-mtp', '--max_track_points', type=int, required=False, default=45,
                    help='maximum points to be tracked for a vehicle')

    ap.add_argument('-pfc', '--path_from_centroid', type=int, required=False, default=1,
                    help='whether to track path from centroid')

    ap.add_argument('-ma', '--max_absent', type=int, required=False, default=5,
                    help='maximum frames a vehicle can be absent, after that it will be deregister')

    ap.add_argument('-m', '--mode', type=str, required=False, default="production",
                    help='execution mode, either `debug` or `production`')

    args = vars(ap.parse_args())

    vt_obj = VehicleTracking(
        args["input"],
        args["detector"],
        args["tracker"],
        args["background"],
        args["max_track_points"],
        args["path_from_centroid"],
        args["max_absent"],
        args["mode"])

    vt_obj.run()
