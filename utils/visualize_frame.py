import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


input_path = "inputs/place5_clip1.mp4"

vidcap = cv2.VideoCapture(input_path)

width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)


def line_intersect(A1, A2, B1, B2):
    Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2 = A1[0], A1[1], A2[0], A2[1], B1[0], B1[1], B2[0], B2[1]
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
 
    return round(x), round(y)


def get_point(pt1, ref_number, direction):
    c1, c2 = pt1, (503, -8)

    if direction == "lower":
        ratio = 3.8
    elif direction == "upper":
        ratio = -3.8

    cx = round(c2[0] + (c1[0]-c2[0]) * ratio)
    cy = round(c2[1] + (c1[1]-c2[1]) * ratio)
    pt2 = (cx, cy)

    if ref_number == 1:
        return line_intersect(pt1,pt2,(30,421), (380,421))
    elif ref_number == 2:
        return line_intersect(pt1,pt2,(50,362),(400,362))
    elif ref_number == 3:
        return line_intersect(pt1,pt2,(110,335), (400,335))
    elif ref_number == 4:
        return line_intersect(pt1,pt2,(145,297), (410,297))


while vidcap.isOpened():
    status, frame = vidcap.read()

    frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    sideway1_coords = (
        np.array(
            [(12, 270), (12, 408), (366, 115), (300, 115)],
            dtype=np.int32,
        ).reshape((-1, 1, 2)),
    )
    cv2.polylines(frame,sideway1_coords,isClosed=True,color=(0, 0, 0),thickness=2,)

    lane1_coords = (
        np.array(
            [(12, 408), (12, 452), (168, 465), (430, 100), (390, 97)],
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
            [(801, 426), (950, 417), (945, 364), (630, 105), (574, 105)],
            dtype=np.int32,
        ).reshape((-1, 1, 2)),
    )
    cv2.polylines(frame,lane4_coords,isClosed=True,color=(0, 0, 0),thickness=2,)

    sideway2_coords = (
        np.array(
            [(945, 364), (935, 270), (720, 128), (658, 128)],
            dtype=np.int32,
        ).reshape((-1, 1, 2)),
    )
    cv2.polylines(frame,sideway2_coords,isClosed=True,color=(0, 0, 0),thickness=2,)

    # lane_ref
    cv2.circle(frame,(5, 350),radius=3,color=(255,0,0),thickness=-1)
    cv2.circle(frame,(56, 497),radius=3,color=(255,0,0),thickness=-1)
    cv2.circle(frame,(245, 510),radius=3,color=(255,0,0),thickness=-1)
    cv2.circle(frame,(757, 523),radius=3,color=(255,0,0),thickness=-1)
    cv2.circle(frame,(930, 506),radius=3,color=(255,0,0),thickness=-1)

    # mid_ref
    cv2.line(frame, (200, 241), (420, 259), (255, 255, 0), 1)
    cv2.line(frame, (300, 171), (445, 183), (255, 255, 0), 1)
    cv2.line(frame, (360, 119), (460, 127), (255, 255, 0), 1)
    cv2.line(frame, (535, 131), (662, 135), (255, 255, 0), 1)
    cv2.line(frame, (550, 187), (733, 190), (255, 255, 0), 1)
    cv2.line(frame, (575, 267), (825, 270), (255, 255, 0), 1)

    # classupdate_line
    # cv2.line(frame, (12, 270), (12, 408), (255, 0, 255), 1)
    # cv2.line(frame, (168,190), (224,231), (255, 0, 255), 2)
    # cv2.line(frame, (12, 452), (348, 483), (255, 0, 255), 2)
    # cv2.line(frame, (178,297), (401,315), (255, 0, 255), 2)
    # cv2.line(frame, (571, 241), (764, 242), (255, 0, 255), 2)
    # cv2.line(frame, (604, 354), (874, 345), (255, 0, 255), 2)
    # cv2.line(frame, (768, 218), (800, 185), (255, 0, 255), 2)
    # cv2.line(frame, (945, 364), (935, 270), (255, 0, 255), 1)

    # deregistering_line_rightdirection
    cv2.line(frame, (395,93), (471,98), (0, 255, 0), 1)
    cv2.line(frame, (644, 488), (844, 484), (0,255,0), 1)
    cv2.line(frame, (810,436), (957,424), (0,255,0), 1)

    # deregistering_line_wrongdirection
    cv2.line(frame, (6,458), (345,494), (0,0,255), 1)
    cv2.line(frame, (529, 100), (613, 100), (0, 0, 255), 1)

    # speed_reflines, lane1, lane2
    cv2.line(frame, (30,421), (380,421), (255, 0, 0), 1)
    cv2.line(frame, (50,362), (400,362), (255, 0, 0), 1)
    cv2.line(frame, (110,335), (400,335), (255, 0, 0), 1)
    cv2.line(frame, (145,297), (410,297), (255, 0, 0), 1)
    cv2.line(frame, (165,278), (420,278), (255, 0, 0), 1)

    # speed_reflines, lane3, lane4
    cv2.line(frame, (570,266), (850,266), (255, 0, 0), 1)
    cv2.line(frame, (570,298), (870,298), (255, 0, 0), 1)
    cv2.line(frame, (570,319), (900,319), (255, 0, 0), 1)
    cv2.line(frame, (600,364), (950,364), (255, 0, 0), 1)
    cv2.line(frame, (600,398), (950,398), (255, 0, 0), 1)

    # coords = [
    #     [(93, 468), (359, 159)],
    #     [(254, 484), (420, 156)],
    #     [(555, 110), (728, 456)],
    #     [(597, 111), (910, 469)],
    #     [(22.4, 444.9), (378.3, 112.9)],
    #     [(339.61, 495.84), (467.45, 102.44)],
    # ]

    # for pt1, pt2 in coords:
    #     angle = math.atan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
    #     # print(angle)
    #     cv2.arrowedLine(frame, tuple(round(p) for p in pt1), tuple(round(p) for p in pt2), color=(0,0,255), thickness=2, tipLength=0.02)

    # pt5 = (259, 335)
    # cv2.circle(frame,pt5,radius=1,color=(0,255,0),thickness=-1)

    # pt1 = coords[4][0]
    # c1, c2 = coords[4]
    # cx = round(c2[0] + (c1[0]-c2[0]) * -3.8)
    # cy = round(c2[1] + (c1[1]-c2[1]) * -3.8)
    # pt2 = (cx, cy)
    # cv2.line(frame, tuple(round(p) for p in pt1), pt2, (0, 0, 255), 1)

    # pt3 = coords[5][0]
    # c1, c2 = coords[5]
    # cx = round(c2[0] + (c1[0]-c2[0]) * -3.8)
    # cy = round(c2[1] + (c1[1]-c2[1]) * -3.8)
    # pt4 = (cx, cy)
    # cv2.line(frame, tuple(round(p) for p in pt3), pt4, (0, 0, 255), 1)

    # pt6 = line_intersect(pt1,pt2,pt3,pt4)# (501, -2)
    # print(pt6)

    # c1, c2 = pt5, pt6
    # cx = round(c2[0] + (c1[0]-c2[0]) * 3.8)
    # cy = round(c2[1] + (c1[1]-c2[1]) * 3.8)
    # pt7 = (cx, cy)
    # pt8 = line_intersect(pt5,pt7,(50,362),(400,362))
    # cv2.line(frame, pt5, pt8, (0,255, 0), 2)

    # for pt in [(343, 380, 1), (143, 380, 1), (255, 342, 2), (250, 314, 3)]:
    #     pt2 = get_point(pt[:2], pt[2], direction="lower")
    #     pt3 = get_point(pt[:2], pt[2]+1, direction="upper")
    #     cv2.line(frame, pt[:2], pt2, (0,255, 0), 1)
    #     cv2.line(frame, pt[:2], pt3, (0,255, 0), 1)
    #     cv2.circle(frame,pt[:2],radius=1,color=(255,0,0),thickness=-1)

    #     pixle_distance1 = distance.euclidean(pt[:2], pt2)
    #     pixle_distance2 = distance.euclidean(pt[:2], pt3)

    #     if pt[2] == 1:
    #         pixles_per_meter = (pixle_distance1 + pixle_distance2) / 6
    #     elif pt[2] == 2:
    #         pixles_per_meter = (pixle_distance1 + pixle_distance2) / 3
    #     elif pt[2] == 3:
    #         pixles_per_meter = (pixle_distance1 + pixle_distance2) / 6

    #     metre_distance1 = pixle_distance1 / pixles_per_meter
    #     metre_distance2 = pixle_distance2 / pixles_per_meter

    #     print(f"pixle_distance1: {round(pixle_distance1, 2)}, metre_distance: {round(metre_distance1, 2)}", end="\n")
        # print(f"pixle_distance2: {round(pixle_distance2, 2)}, metre_distance: {round(metre_distance2, 2)}")

    # cv2.line(frame, (550,266), (850,266), (255, 0, 0), 1)
    # cv2.line(frame, (600,364), (950,364), (255, 0, 0), 1)

    # pt1 = [319.6, 251.3] # bottom
    # pt2 = [350.4, 224.8] # tip

    # pt1 = [375.274, 243.91] # bottom
    # pt2 = [400.636, 216.507] # tip

    # pt1 = [758.94, 364.423] # bottom
    # pt2 = [800.45, 398.181] # tip

    # pt1_copy = pt1.copy()
    # pt2_copy = pt2.copy()

    # cv2.arrowedLine(frame, tuple(round(p) for p in pt1), tuple(round(p) for p in pt2), (255, 0, 0), 2, tipLength=0.25)

    # pt1[1] = -pt1[1]
    # pt2[1] = -pt2[1]
    # angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    # print(angle)

    # lane_angle = math.atan2(-2 - pt1[1], 501 - pt1[0]) - 3.1416
    # print(lane_angle)

    # angle = -lane_angle - angle
    # print(angle)

    # c = math.cos(angle)
    # s = math.sin(angle)
    # pt2[0] -= pt1[0]
    # pt2[1] -= pt1[1]

    # x = [0, 0]
    # x[0] = pt1[0] + c * pt2[0] - s * pt2[1]
    # x[1] = -pt1[1] + s * pt2[0] + c * pt2[1]
    # print(x)

    # cv2.arrowedLine(frame, tuple(round(p) for p in pt1_copy), tuple(round(p) for p in x), (0, 255, 0), 2)

    plt.imshow(frame)
    plt.show()
    break

    # cv2.imshow("", frame)
    # key = cv2.waitKey(-1)
    # if key == ord("q"):
    #     break

vidcap.release()
cv2.destroyAllWindows()
