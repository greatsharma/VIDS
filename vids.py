import os
import cv2
import sys
import time
import argparse
import datetime
import subprocess
import numpy as np

from camera_metadata import CAMERA_METADATA
from detectors import VanillaYoloDetector
from trackers import CentroidTracker, KalmanTracker
from utils import draw_tracked_objects
from utils import init_lane_detector, init_direction_detector, init_classupdate_line
from utils import init_position_wrt_midrefs, init_speed_detector


class VehicleTracking(object):
    def __init__(
        self,
        input_path,
        inference_type,
        output,
        output_fps,
        resize,
        detection_thresh,
        tracker_type,
        max_track_pts,
        max_absent,
        min_continous_presence,
        direction_detector_interval,
        mode,
    ):

        if input_path.startswith("inputs"):
            self.camera_id = input_path.split("/")[1].split("_")[0]
            self.input_path = input_path
        else:
            raise ValueError("Place input videos in inputs folder only, rtsp is not supported yet")

        self.inference_type = inference_type

        self.output = output
        self.output_fps = output_fps            

        self.camera_meta = CAMERA_METADATA[self.camera_id]

        self.resize = resize
        self.detection_thresh = detection_thresh
        self.tracker_type = tracker_type
        self.max_track_pts = max_track_pts
        self.max_absent = max_absent
        self.min_continous_presence = min_continous_presence
        self.direction_detector_interval = direction_detector_interval
        self.mode = mode

        self.speed_detector = init_speed_detector(self.camera_meta)

        self.vidcap = cv2.VideoCapture(self.input_path)

        if self.output_fps is None:
            self.output_fps = int(self.vidcap.get(cv2.CAP_PROP_FPS))

        self._init_detector()
        self._init_tracker()

    def _init_detector(self):
        _, initial_frame = self.vidcap.read()

        if self.resize[0] <= 1:
            self.frame_h, self.frame_w = tuple(
                int(d * s) for d, s in zip(initial_frame.shape[:2], self.resize)
            )
        else:
            self.frame_h, self.frame_w = self.resize

        initial_frame = cv2.resize(
            initial_frame,
            dsize=(self.frame_w, self.frame_h),
            interpolation=cv2.INTER_LINEAR,
        )

        lane_detector = init_lane_detector(self.camera_meta)

        if self.inference_type == "trt":
            self.detector = TrtYoloDetector(
                initial_frame,
                lane_detector,
                self.detection_thresh,
            )
        else:
            self.detector = VanillaYoloDetector(
                initial_frame,
                lane_detector,
                self.detection_thresh,
            )

    def _init_tracker(self):
        lane_detector = init_lane_detector(self.camera_meta)
        direction_detector = init_direction_detector(self.camera_meta)
        classupdate_line = init_classupdate_line(self.camera_meta)
        pos_wrt_midrefs__detector = init_position_wrt_midrefs(self.camera_meta)
        
        initial_maxdistances = self.camera_meta["initial_maxdistances"]

        lane_angles = {
            "1": self.camera_meta["lane1"]["angle"],
            "2": self.camera_meta["lane2"]["angle"],
            "3": self.camera_meta["lane3"]["angle"],
            "4": self.camera_meta["lane4"]["angle"],
            "5": self.camera_meta["lane5"]["angle"],
            "6": self.camera_meta["lane6"]["angle"],
        }

        if self.tracker_type == "centroid":
            self.tracker = CentroidTracker(
                lane_detector,
                direction_detector,
                self.direction_detector_interval,
                initial_maxdistances,
                classupdate_line,
                pos_wrt_midrefs__detector,
                lane_angles,
                self.max_absent,
            )
        else:
            self.tracker = KalmanTracker(
                lane_detector,
                direction_detector,
                self.direction_detector_interval,
                initial_maxdistances,
                classupdate_line,
                pos_wrt_midrefs__detector,
                lane_angles,
                self.max_absent,
            )

    def _compress_video(self, input_path, output_path, del_prev_video):
        status = subprocess.call(
            [
                "ffmpeg",
                "-i",
                input_path,
                "-vcodec",
                "libx264",
                "-crf",
                "30",
                output_path,
                "-hide_banner",
                "-loglevel",
                "panic",
                "-y",
            ]
        )

        if status:
            msg = f"Compression_Error : {datetime.datetime.now()} : Unable to compress {input_path} !"
            self.error_filewriter.write(msg + "\n")
            print(msg)
        else:
            if del_prev_video:
                os.remove(input_path)

    def _clean_exit(self):
        print(
            "\nExecuting clean exit, this may take few minutes depending on compression...\n"
        )

        if self.output:
            self.videowriter.release()
            compressed_filename = self.video_filename.split(".")[0] + "_comp.avi"
            self._compress_video(self.video_filename, compressed_filename, True)

    def run(self):
        self.frame_count = 0

        date = datetime.datetime.now()
        date = date.strftime("%d_%m_%Y_%H:%M:%S")
        date = date.replace(":", "")

        curr_folder = "outputs/place5/" + date

        if not os.path.exists(curr_folder):
            os.mkdir(curr_folder)

        if self.output:
            self.video_filename = curr_folder + "/vids.avi"
            self.videowriter = cv2.VideoWriter(
                self.video_filename,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                self.output_fps,
                (960, 540),
            )

        tik1 = time.time()

        while self.vidcap.isOpened():
            tik2 = time.time()
            status, frame = self.vidcap.read()

            while not status:
                if self.input_path.startswith("inputs"):
                    self._clean_exit()
                    self.vidcap.release()
                    cv2.destroyAllWindows()
                    sys.exit()

                print("Waiting for frame!!")
                self.vidcap.release()
                time.sleep(1)
                self.vidcap = cv2.VideoCapture(self.input_path)
                status, frame = self.vidcap.read()

            self.frame_count += 1

            frame = cv2.resize(
                frame,
                dsize=(self.frame_w, self.frame_h),
                interpolation=cv2.INTER_LINEAR,
            )

            detection_list, ped_and_cattles_list = self.detector.detect(frame)
            tracked_objects = self.tracker.update(detection_list)

            for rect in ped_and_cattles_list:
                cv2.rectangle(frame, rect[:2], rect[2:], (255, 0, 255), 2)

            draw_tracked_objects(self, frame, tracked_objects)

            if self.mode == "debug":
                for l in [1,2,3,4]:
                    cv2.polylines(frame, [self.camera_meta[f"lane{l}"]["lane_coords"]],
                        isClosed=True, color=(0, 0, 0), thickness=1,
                    )
                    cv2.circle(frame, self.camera_meta[f"lane{l}"]["lane_ref"],
                        radius=3, color=(0, 0, 255), thickness=-1,
                    )

                    pt1, pt2 = self.camera_meta[f"lane{l}"]["deregistering_line_rightdirection"]
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 1)

                    pt1, pt2 = self.camera_meta[f"lane{l}"]["deregistering_line_wrongdirection"]
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

                    for ref in self.camera_meta[f"lane{l}"]["speed_reflines"]:
                        cv2.line(frame, ref[0], ref[1], (255, 0, 255), 1)

            if self.output:
                self.videowriter.write(frame)

            cv2.imshow(f"VIDS", frame)
            key = cv2.waitKey(1)
            
            if key == ord("p"):
                cv2.waitKey(-1)
            elif key == ord("q"):
                break

            tok = time.time()
            curr_fps = round(1.0 / (tok-tik2), 4)
            avg_fps = round(self.frame_count / (tok-tik1), 4)
            print(self.frame_count, curr_fps, avg_fps)

        if self.output:
            self._clean_exit()
    
        self.vidcap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        default="inputs/place5_clip1.avi",
        help="path to input video",
    )

    ap.add_argument(
        "-if",
        "--inference",
        type=str,
        required=False,
        default="vanilla",
        choices=["vanilla", "trt"],
        help="type pf inference",
    )

    ap.add_argument(
        "-o",
        "--output",
        type=int,
        required=False,
        default=0,
        help="whether to write output videos, default is 0",
    )

    ap.add_argument(
        "-ofps",
        "--output_fps",
        type=int,
        required=False,
        help="output fps, default is fps of input video",
    )

    ap.add_argument(
        "-r",
        "--resize",
        nargs="+",
        type=float,
        required=False,
        default=[0.5, 0.5],
        help="resize factor/shape of image",
    )

    ap.add_argument(
        "-dt",
        "--detection_thresh",
        type=float,
        required=False,
        default=0.5,
        help="detection threshold",
    )

    ap.add_argument(
        "-t",
        "--tracker",
        type=str,
        required=False,
        default="kalman",
        help="tracker to use",
        choices=["centroid", "kalman"],
    )

    ap.add_argument(
        "-mtp",
        "--max_track_points",
        type=int,
        required=False,
        default=40,
        help="maximum points to be tracked for a vehicle",
    )

    ap.add_argument(
        "-ma",
        "--max_absent",
        type=int,
        required=False,
        default=20,
        help="maximum frames a vehicle can be absent, after that it will be deregistered",
    )

    ap.add_argument(
        "-mcp",
        "--min_continous_presence",
        type=int,
        required=False,
        default=3,
        help="minimum continous frames a vehicle is present, if less then this, then it will be deregistered",
    )

    ap.add_argument(
        "-ddi",
        "--direction_detector_interval",
        type=int,
        required=False,
        default=16,
        help="interval between two frames for direction detection",
    )

    ap.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        default="release",
        help="execution mode, either `debug`, `release`, `pretty`",
    )

    args = vars(ap.parse_args())

    if args["inference"] == "trt":
        from detectors.trt_detector import TrtYoloDetector

    vt_obj = VehicleTracking(
        args["input"],
        args["inference"],
        args["output"],
        args["output_fps"],
        args["resize"],
        args["detection_thresh"],
        args["tracker"],
        args["max_track_points"],
        args["max_absent"],
        args["min_continous_presence"],
        args["direction_detector_interval"],
        args["mode"],
    )

    print("\n")
    vt_obj.run()
