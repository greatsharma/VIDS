import os
import cv2
import sys
import time
import argparse
import datetime
import threading
import subprocess
import numpy as np

from camera_metadata import CAMERA_METADATA
from detectors import VanillaYoloDetector
from trackers import CentroidTracker, KalmanTracker
from utils import draw_text_with_backgroud, draw_tracked_objects
from utils import init_lane_detector, init_direction_detector, init_classupdate_line


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
        self.mode = mode

        # self.countintervals = self.camera_meta["adaptive_countintervals"]

        self.vidcap = cv2.VideoCapture(self.input_path)

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

        self.img_for_text = np.zeros(shape=(self.frame_h, 360))

        self.lane_detector = init_lane_detector(self.camera_meta)

        if self.inference_type == "trt":
            self.detector = TrtYoloDetector(
                initial_frame,
                self.lane_detector,
                self.detection_thresh,
            )
        else:
            self.detector = VanillaYoloDetector(
                initial_frame,
                self.lane_detector,
                self.detection_thresh,
            )

    def _init_tracker(self):
        direction_detector = init_direction_detector(self.camera_meta)
        classupdate_line = init_classupdate_line(self.camera_meta)

        initial_maxdistances = self.camera_meta["initial_maxdistances"]

        lane_angles = {
            "1": self.camera_meta["lane1"]["angle"],
            "2": self.camera_meta["lane2"]["angle"],
            "3": self.camera_meta["lane3"]["angle"],
            "4": self.camera_meta["lane4"]["angle"],
        }
        
        velocity_regression = {} #self.camera_meta["velocity_regression"]

        if self.tracker_type == "centroid":
            self.tracker = CentroidTracker(
                direction_detector,
                initial_maxdistances,
                classupdate_line,
                lane_angles,
                velocity_regression,
                self.max_absent,
            )
        else:
            self.tracker = KalmanTracker(
                direction_detector,
                initial_maxdistances,
                classupdate_line,
                lane_angles,
                velocity_regression,
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

    def _delete_oneday_videos(self, output_path):
        status = subprocess.call(
            [
                "python",
                "delete_oneday_videos.py",
                "-ip",
                output_path,
            ]
        )

        if status:
            msg = f"VideoDeletion_Error : {datetime.datetime.now()} : Error in function _delete_oneday_videos !"
            print(msg)

    def _clean_exit(self):
        print(
            "\nExecuting clean exit, this may take few minutes depending on compression...\n"
        )

        if self.output:
            self.videowriter.release()
            compressed_filename = self.video_filename.split(".")[0] + "_comp.avi"
            self._compress_video(self.video_filename, compressed_filename, True)

    def run(self):
        frame_count = 0

        date = datetime.datetime.now()
        videodeletion_day = date

        videodeletion_initialization_day = videodeletion_day + datetime.timedelta(
            days=4
        )
        videodeletion_initialization_day = videodeletion_initialization_day.strftime(
            "%d_%m_%Y"
        )
        flag_videodeletion = False

        date = date.strftime("%d_%m_%Y_%H:%M:%S")

        currentday_dir = f"outputs/{self.camera_id}/{date[:10]}/"
        if not os.path.exists(currentday_dir):
            os.mkdir(currentday_dir)

        currenthour_dir = currentday_dir + date[11:13] + "/"
        if not os.path.exists(currenthour_dir):
            os.mkdir(currenthour_dir)

        log_filename = currenthour_dir + date[11:13] + ".txt"
        self.log_filewriter = open(log_filename, "w")

        trackpath_filename = currenthour_dir + date[11:13] + f"_trkpath.txt"
        self.tracker.trackpath_filewriter = open(trackpath_filename, "w")

        cc_filename = currenthour_dir + date[11:13] + "_finalcounts.txt"
        self.cc_filewriter = open(cc_filename, "w")

        if self.output:
            self.video_filename = currenthour_dir + date[11:13] + ".avi"
            self.videowriter = cv2.VideoWriter(
                self.video_filename,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                self.output_fps,
                (1784, 540),
            )

        flag1 = True
        flag2 = True

        tik1 = time.time()

        while self.vidcap.isOpened():
            tik2 = time.time()
            status, frame = self.vidcap.read()

            while not status:
                if self.input_path.startswith("inputs"):
                    self._clean_exit(currenthour_dir, currentday_dir)
                    self.vidcap.release()
                    cv2.destroyAllWindows()
                    sys.exit()

                print("Waiting for frame!!")
                self.vidcap.release()
                time.sleep(1)
                self.vidcap = cv2.VideoCapture(self.input_path)
                status, frame = self.vidcap.read()

            frame_count += 1

            date = datetime.datetime.now()
            date = date.strftime("%d_%m_%Y_%H:%M:%S")

            if videodeletion_initialization_day == date[:10]:
                flag_videodeletion = True

            if date[11:16] == "00:00":
                if flag1:
                    if self.output and flag_videodeletion:
                        self.videowriter.release()

                        output_path = f"outputs/{self.camera_id}/{videodeletion_day.strftime('%d_%m_%Y')}/"
                        t2 = threading.Thread(
                            target=self._delete_oneday_videos,
                            kwargs={"output_path": output_path},
                        )
                        t2.start()

                        videodeletion_day += datetime.timedelta(days=1)

                    currentday_dir = f"outputs/{self.camera_id}/{date[:10]}/"
                    if not os.path.exists(currentday_dir):
                        os.mkdir(currentday_dir)

                    self.tracker.next_objid = 0
                    self.logged_ids = []

                    flag1 = False

            else:
                flag1 = True

            if date[14:16] == "00":
                if flag2:
                    currenthour_dir = currentday_dir + date[11:13] + "/"
                    if not os.path.exists(currenthour_dir):
                        os.mkdir(currenthour_dir)

                    log_filename = currenthour_dir + date[11:13] + ".txt"
                    self.log_filewriter = open(log_filename, "w")

                    trackpath_filename = currenthour_dir + date[11:13] + f"_trkpath.txt"
                    self.tracker.trackpath_filewriter = open(trackpath_filename, "w")

                    cc_filename = currenthour_dir + date[11:13] + "_finalcounts.txt"
                    self.cc_filewriter = open(cc_filename, "w")

                    print("\n-------cleared class counters-------")
                    print(f"-------initialized new file for the hour-------{date}\n")

                    if self.output:
                        compressed_file_name = (
                            self.video_filename.split(".")[0] + "_comp.avi"
                        )
                        t4 = threading.Thread(
                            target=self._compress_video,
                            args=(self.video_filename, compressed_file_name, True),
                        )
                        t4.start()

                        self.video_filename = currenthour_dir + date[11:13] + ".avi"
                        self.videowriter = cv2.VideoWriter(
                            self.video_filename,
                            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                            15.0,
                            (1784, 540),
                        )

                        print("-------initialized new video for the hour-------\n")

                    flag2 = False
            else:
                flag2 = True

            frame = cv2.resize(
                frame,
                dsize=(self.frame_w, self.frame_h),
                interpolation=cv2.INTER_LINEAR,
            )

            detection_list = self.detector.detect(frame)
            tracked_objects = self.tracker.update(detection_list)

            # self._count_vehicles(tracked_objects)

            # self._log(tracked_objects)

            draw_tracked_objects(self, frame, tracked_objects)

            # if self.mode == "debug":
            #     for l in ["leftlane", "rightlane"]:
            #         cv2.polylines(
            #             frame,
            #             [self.camera_meta[f"{l}_coords"]],
            #             isClosed=True,
            #             color=(0, 0, 0),
            #             thickness=1,
            #         )
            #         cv2.circle(
            #             frame,
            #             self.camera_meta[f"{l}_ref"],
            #             radius=3,
            #             color=(0, 0, 255),
            #             thickness=-1,
            #         )

                # pt1,pt2 = self.camera_meta["mid_ref"]
                # cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

                # pt1, pt2, pt3, pt4 = self.camera_meta["adaptive_countintervals"][
                #     "3t,4t,5t,6t,lgv,tractr,2t,bus,mb"
                # ]
                # cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
                # cv2.line(frame, pt3, pt4, (255, 255, 255), 1)

                # pt1, pt2, pt3, pt4 = self.camera_meta["adaptive_countintervals"][
                #     "wrong_direction"
                # ]
                # cv2.line(frame, pt1, pt2, (255, 153, 153), 1)
                # cv2.line(frame, pt3, pt4, (255, 153, 153), 1)

                # pt1, pt2 = self.camera_meta["classupdate_line"]
                # cv2.line(frame, pt1, pt2, (255, 0, 255), 1)

                # pt1, pt2 = self.camera_meta["deregistering_line_rightdirection"]
                # cv2.line(frame, pt1, pt2, (255, 255, 0), 1)

                # pt1, pt2 = self.camera_meta["deregistering_line_wrongdirection"]
                # cv2.line(frame, pt1, pt2, (255, 255, 0), 1)

                # for name, xcoord in zip(
                #     ["Class", "L1", "L2", "Total"], [15, 170, 240, 300]
                # ):
                #     draw_text_with_backgroud(
                #         self.img_for_text,
                #         name,
                #         x=xcoord,
                #         y=150,
                #         font_scale=0.6,
                #         thickness=2,
                #     )

                # y = 180
                # vehicles_lane1 = 0
                # vehicles_lane2 = 0

                # for (k1, v1), (_, v2) in zip(
                #     self.class_counts["1"].items(), self.class_counts["2"].items()
                # ):
                #     vehicles_lane1 += v1
                #     vehicles_lane2 += v2

                #     for name, xcoord, bg in zip(
                #         [k1, str(v1), str(v2), str(v1 + v2)],
                #         [15, 170, 240, 310],
                #         (None, (246, 231, 215), (242, 226, 209), (241, 222, 201)),
                #     ):

                #         draw_text_with_backgroud(
                #             self.img_for_text,
                #             name,
                #             x=xcoord,
                #             y=y,
                #             font_scale=0.5,
                #             thickness=1,
                #             background=bg,
                #         )

                #     y += 20

                # y += 20
                # for name, xcoord, bg in zip(
                #     [
                #         "Total",
                #         str(vehicles_lane1),
                #         str(vehicles_lane2),
                #         str(vehicles_lane1 + vehicles_lane2),
                #     ],
                #     [15, 170, 240, 310],
                #     (None, (246, 231, 215), (242, 226, 209), (241, 222, 201)),
                # ):

                #     draw_text_with_backgroud(
                #         self.img_for_text,
                #         name,
                #         x=xcoord,
                #         y=y,
                #         font_scale=0.55,
                #         thickness=2,
                #         background=bg,
                #     )

                # draw_text_with_backgroud(
                #     self.img_for_text,
                #     f"WD : {self.wrongdir_count}",
                #     x=15,
                #     y=500,
                #     font_scale=0.6,
                #     thickness=2,
                #     background=(242, 226, 209),
                # )

                # if self.mode != "debug":
                #     for name in ["L1", "L2"]:
                #         k = name + " annotation_data"
                #         txt = name + " : "

                #         if name == "L1":
                #             txt += str(vehicles_lane1)
                #         else:
                #             txt += str(vehicles_lane2)

                #         cv2.line(
                #             frame,
                #             self.camera_meta[k][0],
                #             self.camera_meta[k][1],
                #             (128, 0, 128),
                #             2,
                #         )
                #         cv2.line(
                #             frame,
                #             self.camera_meta[k][1],
                #             self.camera_meta[k][2],
                #             (128, 0, 128),
                #             2,
                #         )
                #         draw_text_with_backgroud(
                #             frame,
                #             txt,
                #             x=self.camera_meta[k][3],
                #             y=self.camera_meta[k][4],
                #             font_scale=0.7,
                #             thickness=1,
                #             background=(128, 0, 128),
                #             foreground=(255, 255, 255),
                #             box_coords_1=(-7, 7),
                #             box_coords_2=(10, -10),
                #         )

            # out_frame = np.hstack((frame, self.img_for_text))
            cv2.imshow(f"VIDS", frame)

            if self.output:
                self.videowriter.write(frame)

            key = cv2.waitKey(1)

            if key == ord("p"):
                cv2.waitKey(-1)
            elif key == ord("q"):
                break

            print(frame_count)

        self._clean_exit(currenthour_dir, currentday_dir)
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
        required=True,
        help="output fps",
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
        default=15,
        help="maximum points to be tracked for a vehicle",
    )

    ap.add_argument(
        "-ma",
        "--max_absent",
        type=int,
        required=False,
        default=10,
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
        args["mode"],
    )

    print("\n")
    vt_obj.run()
