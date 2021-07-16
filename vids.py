import os
import cv2
import sys
import time
import argparse
import datetime
import threading
import subprocess
import numpy as np
from collections import deque
from waitress import serve
from flask import Flask, Response, render_template, request

from camera_metadata import CAMERA_METADATA
from trackers import CentroidTracker, KalmanTracker
from utils import draw_tracked_objects, draw_text_with_backgroud
from utils import init_lane_detector, init_direction_detector, init_classupdate_line
from utils import init_position_wrt_midrefs, init_speed_detector


app = Flask(__name__)


class VIDS(object):
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
    ):

        self.pause_stream = False
        self.stop_stream = False

        if input_path.startswith("inputs"):
            self.camera_id = input_path.split("/")[1].split("_")[0]
            self.input_path = input_path
        else:
            raise ValueError("Place input videos in inputs folder only, rtsp is not supported yet")

        self.inference_type = inference_type

        self.output = output
        self.output_fps = output_fps            

        self.camera_meta = CAMERA_METADATA[self.camera_id]

        if not os.path.exists(f"outputs/{self.camera_id}"):
            os.mkdir(f"outputs/{self.camera_id}/")
        
        self.resize = resize
        self.detection_thresh = detection_thresh
        self.tracker_type = tracker_type
        self.max_track_pts = max_track_pts
        self.max_absent = max_absent
        self.min_continous_presence = min_continous_presence
        self.direction_detector_interval = direction_detector_interval

        self.speed_detector = init_speed_detector(self.camera_meta)

        self.vidcap = cv2.VideoCapture(self.input_path)

        if self.output_fps is None:
            self.output_fps = int(self.vidcap.get(cv2.CAP_PROP_FPS))

        self.logbuffer_length = 15
        self.logged_ids = []
        self.log_buffer = deque([])

        self._init_tracker()

        self.lock = threading.Lock()

    def _init_detector(self):
        _, initial_frame = self.vidcap.read()

        if self.resize[0] <= 1:
            self.frame_h, self.frame_w = tuple(
                int(d * s) for d, s in zip(initial_frame.shape[:2], self.resize)
            )
        else:
            self.frame_h, self.frame_w = self.resize

        self.img_for_log = np.zeros(
            (self.frame_h, int(self.frame_w / 3.75), 3), dtype=np.uint8
        )
        self.img_for_log[:, :, 0:3] = (243, 227, 218)

        initial_frame = cv2.resize(
            initial_frame,
            dsize=(self.frame_w, self.frame_h),
            interpolation=cv2.INTER_LINEAR,
        )

        lane_detector = init_lane_detector(self.camera_meta)

        if self.inference_type == "trt":
            from detectors.trt_detector import TrtYoloDetector
            self.detector = TrtYoloDetector(
                initial_frame,
                lane_detector,
                self.detection_thresh,
            )
        else:
            from detectors.yolo_detector import VanillaYoloDetector
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

        intesection_point_of_lanes = self.camera_meta["intesection_point_of_all_lanes"]

        if self.tracker_type == "centroid":
            self.tracker = CentroidTracker(
                lane_detector,
                direction_detector,
                self.direction_detector_interval,
                initial_maxdistances,
                classupdate_line,
                pos_wrt_midrefs__detector,
                lane_angles,
                intesection_point_of_lanes,
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
                intesection_point_of_lanes,
                self.max_absent,
            )

    def _log(self, tracked_objs):
        lane_counts = {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0
        }

        for obj in tracked_objs.values():
            if obj.lane in ["5", "6"]:
                continue

            obj_bottom = (
                obj.obj_bottom
                if self.tracker_type == "centroid"
                else (obj.state[0], obj.state[2])
            )

            lane_counts[obj.lane] += 1

            if obj.lane in ["1", "2"]:
                (A1x, A1y), (B1x, B1y) = self.camera_meta["lane" + obj.lane]["speed_reflines"][0]
                (A2x, A2y), (B2x, B2y) = self.camera_meta["lane" + obj.lane]["speed_reflines"][3]
            else:
                (A1x, A1y), (B1x, B1y) = self.camera_meta["lane" + obj.lane]["speed_reflines"][3]
                (A2x, A2y), (B2x, B2y) = self.camera_meta["lane" + obj.lane]["speed_reflines"][0]

            Px, Py = obj_bottom
            position1 = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)
            position2 = (Px - A2x) * (B2y - A2y) - (Py - A2y) * (B2x - A2x)

            if position1 > 0 and position2 < 0 and obj.starttime is None:
                obj.starttime = datetime.datetime.now()

            elif (
                (position1 < 0 or position2 > 0)
                and obj.endtime is None
                and obj.starttime is not None
            ):
                obj.endtime = datetime.datetime.now()

            if (
                (obj.starttime is not None)
                and (obj.endtime is not None)
                and (obj.objid not in self.logged_ids)
            ):

                obj_lane = f"lane {obj.lane}"
                obj_class = obj.obj_class[0]
                obj_time = obj.starttime.strftime("%Y:%m:%d:%H:%M:%S")[11:]
                obj_direction = obj.direction
                obj_speed = str(obj.instspeed_list[-1])
                if obj_direction == "wrong" or obj_speed == "0":
                    obj_speed = ""

                log_tuple = (obj_lane, obj_class, obj_time, obj_speed, obj_direction)

                if len(self.log_buffer) >= self.logbuffer_length:
                    self.log_buffer.rotate(-1)
                    self.log_buffer[self.logbuffer_length-1] = log_tuple
                else:
                    self.log_buffer.append(log_tuple)

                self.logged_ids.append(obj.objid)

        self.img_for_log[:, :, 0:3] = (243, 227, 218)

        for name, xcoord in zip(["Lane", "Class", "Time", "kmph"], [5, 60, 130, 195]):
            draw_text_with_backgroud(
                self.img_for_log,
                name,
                x=xcoord,
                y=80,
                font_scale=0.5,
                thickness=2,
                background=None,
                foreground=(10, 10, 10),
            )

        y = 110
        for row in self.log_buffer:
            foreground=(0, 100, 0)
            if row[-1] == "wrong":
                foreground=(0, 0, 255)

            for col, xcoord in zip(row, [5, 70, 120, 200]):
                draw_text_with_backgroud(
                    self.img_for_log,
                    col,
                    x=xcoord,
                    y=y,
                    font_scale=0.4,
                    thickness=1,
                    background=None,
                    foreground=foreground
                )
            y += 20

        draw_text_with_backgroud(
            self.img_for_log,
            "Lanes Congestion",
            x=5, y=440,
            font_scale=0.5,
            thickness=2,
            background=None,
            foreground=(10, 10, 10),
            )

        for txt, (xcoord, ycoord) in zip(["Lane 1+2 - ", "Lane 3+4 - "], [(5, 470), (5, 500)]):
            draw_text_with_backgroud(
                self.img_for_log,
                txt,
                x=xcoord,
                y=ycoord,
                font_scale=0.4,
                thickness=1,
                background=None,
                foreground=(10, 10, 10),
            )

            idx = (
                1
                if txt == "Lane 1+2 - "  else
                3              
            )

            total_vehicle = lane_counts[str(idx)] + lane_counts[str(idx+1)]

            foreground = (0, 100, 0)
            txt = "light"

            if total_vehicle > 10:
                foreground = (0, 0, 255)
                txt = "heavy"
            elif total_vehicle > 5:
                foreground = (0, 140, 255)
                txt = "medium"

            draw_text_with_backgroud(
                self.img_for_log,
                txt,
                x=xcoord + 100,
                y=ycoord,
                font_scale=0.4,
                thickness=1,
                background=None,
                foreground=foreground,
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
        self._init_detector()
        self.frame_count = 0
        self.out_frame = None

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
                (1216, 540),
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

            self._log(tracked_objects)

            for rect in ped_and_cattles_list:
                rect = tuple(round(r) for r in rect)
                cv2.rectangle(frame, rect[:2], rect[2:], (255, 0, 255), 2)

            draw_tracked_objects(self, frame, tracked_objects)

            # for l in [1,2,3,4]:
            #     cv2.polylines(frame, [self.camera_meta[f"lane{l}"]["lane_coords"]],
            #         isClosed=True, color=(0, 0, 0), thickness=1,
            #     )
            #     cv2.circle(frame, self.camera_meta[f"lane{l}"]["lane_ref"],
            #         radius=3, color=(0, 0, 255), thickness=-1,
            #     )

            #     pt1, pt2 = self.camera_meta[f"lane{l}"]["deregistering_line_rightdirection"]
            #     cv2.line(frame, pt1, pt2, (255, 255, 0), 1)

            #     pt1, pt2 = self.camera_meta[f"lane{l}"]["deregistering_line_wrongdirection"]
            #     cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

            #     for ref in self.camera_meta[f"lane{l}"]["speed_reflines"]:
            #         cv2.line(frame, ref[0], ref[1], (255, 0, 255), 1)

            draw_text_with_backgroud(
                self.img_for_log,
                "VIDS",
                x=100, y=30,
                font_scale=0.8,
                thickness=2,
                background=None,
                foreground=(10, 10, 10),
                )

            self.out_frame = np.hstack((frame, self.img_for_log))
            if self.output:
                self.videowriter.write(self.out_frame)

            # cv2.imshow("VIDS", self.out_frame)
            # key = cv2.waitKey(1)
            
            # if key == ord("p"):
            #     cv2.waitKey(-1)
            # elif key == ord("q"):
            #     break

            tok = time.time()
            curr_fps = round(1.0 / (tok-tik2), 2)
            avg_fps = round(self.frame_count / (tok-tik1), 1)

            print(f"fc: {self.frame_count}, ifps: {curr_fps}, afps: {avg_fps}")

            if self.pause_stream:
                print("Stream Paused")
                while True:
                    if not self.pause_stream:
                        break
            
            if self.stop_stream:
                break
    
        if self.output:
            self._clean_exit()
    
        self.vidcap.release()
        cv2.destroyAllWindows()

    def gen_frame(self):
        while True:
            with self.lock:
                success, buffer = cv2.imencode(".jpg", self.out_frame)
                if not success:
                    break
                frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


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
        default="trt",
        choices=["vanilla", "trt"],
        help="type of inference",
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
        "-ip",
        type=str,
        required=False,
        default="127.0.0.1"
    )

    ap.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        default=5000
    )

    args = vars(ap.parse_args())

    if not os.path.exists("outputs/"):
        os.mkdir("outputs")

    vids_obj = VIDS(
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
    )

    print("\n")
    # vids_obj.run()
    t = threading.Thread(target=vids_obj.run)
    t.daemon = True
    t.start()

    @app.route("/", methods=["GET","POST"])
    def index():
        if request.method == "POST":
            if 'pause-button' in request.form:
                vids_obj.pause_stream = True
            elif 'play-button' in request.form:
                vids_obj.pause_stream = False
            elif 'stop-button' in request.form:
                vids_obj.stop_stream = True
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(vids_obj.gen_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

    serve(app, host=args["ip"], port=args["port"])
