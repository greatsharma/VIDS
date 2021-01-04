import cv2
from typing import Callable


def intersection_over_union(rect1: list, rect2: list) -> float:
    intersection_x1 = max(rect1[0], rect2[0])
    intersection_y1 = max(rect1[1], rect2[1])
    intersection_x2 = min(rect1[2], rect2[2])
    intersection_y2 = min(rect1[3], rect2[3])

    rect1_area = (rect1[2]-rect1[0]) * (rect1[3]-rect1[1])
    rect2_area = (rect2[2]-rect2[0]) * (rect2[3]-rect2[1])
    intersection_area = min((intersection_x2-intersection_x1), 0) * min((intersection_y2-intersection_y1), 0)

    return intersection_area / (rect1_area + rect2_area - intersection_area + 1e-7)


def intersection_over_rect(rect1: list, rect2: list) -> float:
    # finds ratio of intersection and rect2 (which is smaller in our case)

    if (rect2[0] >= rect1[0]) and (rect2[1] >= rect1[1]) and (rect2[2] <= rect1[2]) and (rect2[3] <= rect1[3]):
        # if rect2 is completely inside rect1
        return 1
    else:
        intersection_x1 = max(rect1[0], rect2[0])
        intersection_y1 = max(rect1[1], rect2[1])
        intersection_x2 = min(rect1[2], rect2[2])
        intersection_y2 = min(rect1[3], rect2[3])

        rect2_area = (rect2[2]-rect2[0]) * (rect2[3]-rect2[1])
        intersection_area = min((intersection_x2-intersection_x1), 0) * min((intersection_y2-intersection_y1), 0)

        return intersection_area / (rect2_area + 1e-7)


def nonmax_suppression(rects: list, iou_thresh=0.3) -> list:
    nms_rects = []
    rects = sorted(rects, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)

    while rects:
        parent_rect = rects.pop(0)

        idx = 0
        for rect in rects:
            if intersection_over_rect(parent_rect, rect) > 0.8 or intersection_over_union(parent_rect, rect) > iou_thresh:
                rects.pop(idx)
            idx += 1

        nms_rects.append(parent_rect)

    return nms_rects


def rects_from_contours(contours: list, is_valid_cntrarea: float, iou_thresh=0.3) -> list:
    rects = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        x1, y1, x2, y2 = x, y, x+w, y+h
        if is_valid_cntrarea(x1, y1, x2, y2):
            rects.append((x1, y1, x2, y2))

    return nonmax_suppression(rects, iou_thresh)


def init_adpative_cntrarea(camera_meta: dict) -> Callable:
    cntrarea_thresh1 = camera_meta["cntrarea_thresh1"]
    cntrarea_thresh2 = camera_meta["cntrarea_thresh2"]
    mid_ref = camera_meta["mid_ref"]

    def is_valid_cntr(x1, y1, x2, y2):
        cntr_area = (x2-x1) * (y2-y1)
        centroid = (x1+x2)//2, (y1+y2)//2

        return (
            cntr_area >= cntrarea_thresh1 and centroid[1] > mid_ref[1] or
            cntr_area >= cntrarea_thresh2 and centroid[1] <= mid_ref[1]
        )

    return is_valid_cntr
