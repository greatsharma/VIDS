import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


input_path = "inputs/place5_clip1.mp4"

vidcap = cv2.VideoCapture(input_path)

width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

while vidcap.isOpened():
    status, frame = vidcap.read()

    frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lane1_coords = (
        np.array(
            [(12, 452), (168, 465), (430, 100), (394, 97)],
            dtype=np.int32,
        ).reshape((-1, 1, 2)),
    )
    cv2.polylines(frame,lane1_coords,isClosed=True,color=(0, 0, 0),thickness=2,)

    lane2_coords = (
        np.array(
            [(168, 465), (348, 483), (469, 102), (430, 100)],
            dtype=np.int32,
        ).reshape((-1, 1, 2)),
    )
    cv2.polylines(frame,lane2_coords,isClosed=True,color=(0, 0, 0),thickness=2,)

    lane3_coords = (
        np.array(
            [(641, 477), (834,473), (574, 105), (530, 105)],
            dtype=np.int32,
        ).reshape((-1, 1, 2)),
    )
    cv2.polylines(frame,lane3_coords,isClosed=True,color=(0, 0, 0),thickness=2,)

    lane4_coords = (
        np.array(
            [(801, 426), (950, 417), (615, 105), (574, 105)],
            dtype=np.int32,
        ).reshape((-1, 1, 2)),
    )
    cv2.polylines(frame,lane4_coords,isClosed=True,color=(0, 0, 0),thickness=2,)

    cv2.circle(frame,(56, 497),radius=3,color=(255,0,0),thickness=-1)
    cv2.circle(frame,(245, 510),radius=3,color=(255,0,0),thickness=-1)
    cv2.circle(frame,(757, 523),radius=3,color=(255,0,0),thickness=-1)
    cv2.circle(frame,(930, 506),radius=3,color=(255,0,0),thickness=-1)

    cv2.line(frame, (86,380), (372,403), (255, 255, 255), 2)
    cv2.line(frame, (184,289), (404,304), (255, 255, 255), 2)
    cv2.line(frame, (608,366), (887,359), (255, 255, 255), 2)
    cv2.line(frame, (577,267), (788,265), (255, 255, 255), 2)

    cv2.line(frame, (12, 452), (348, 483), (255, 0, 255), 2)
    cv2.line(frame, (178,297), (401,315), (255, 0, 255), 2)
    cv2.line(frame, (571, 241), (764, 242), (255, 0, 255), 2)
    cv2.line(frame, (604, 354), (874, 345), (255, 0, 255), 2)

    cv2.line(frame, (262,218), (427,232), (255, 255, 0), 2)
    cv2.line(frame, (565, 220), (745, 220), (255, 255, 0), 2)

    cv2.line(frame, (395,93), (471,98), (0, 255, 0), 1)
    cv2.line(frame, (529, 100), (613, 100), (0, 255, 0), 1)

    cv2.line(frame, (6,458), (345,494), (0,0,255), 1)
    cv2.line(frame, (644, 488), (844, 484), (0,0,255), 1)
    cv2.line(frame, (810,436), (957,424), (0,0,255), 1)

    pt1, pt2 = (93, 468), (359, 159)
    angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
    print(angle)
    cv2.arrowedLine(frame, pt1, pt2, color=(0,0,255), thickness=2, tipLength=0.03)

    pt1, pt2 = (254, 484), (420, 156)
    angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
    print(angle)
    cv2.arrowedLine(frame, pt1, pt2, color=(0,0,255), thickness=2, tipLength=0.03)

    pt1, pt2 = (555, 110), (728, 456)
    angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
    print(angle)
    cv2.arrowedLine(frame, pt1, pt2, color=(0,0,255), thickness=2, tipLength=0.03)

    pt1, pt2 = (597, 111), (910, 469)
    angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
    print(angle)
    cv2.arrowedLine(frame, pt1, pt2, color=(0,0,255), thickness=2, tipLength=0.03)

    plt.imshow(frame)
    plt.show()
    break

vidcap.release()
cv2.destroyAllWindows()
