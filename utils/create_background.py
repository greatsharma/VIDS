import cv2
import argparse
import numpy as np


ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input', type=str, required=True,
                help='path to input video')

ap.add_argument('-o', '--output', type=str, required=True,
                help='path to output image')

ap.add_argument('-fc', '--frame_count', type=int, required=False,
                default=200, help='number of frames to average')

args = vars(ap.parse_args())

frames = []
vidcap = cv2.VideoCapture(args["input"])

_, frame = vidcap.read()

max_frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_ids = np.random.randint(1, max_frame_count, args["frame_count"])

for fid in frame_ids:
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    _, curr_frame = vidcap.read()
    frames.append(curr_frame)

avg_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
del frames

cv2.imwrite(args["output"], avg_frame)
