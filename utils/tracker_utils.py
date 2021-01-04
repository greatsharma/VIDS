import cv2
from typing import Callable
from scipy.spatial import distance


def init_lane_detector(camera_meta: dict) -> Callable:
    leftlane_coords = camera_meta["leftlane_coords"]
    rightlane_coords = camera_meta["rightlane_coords"]

    def lane_detector(centroid):
        if cv2.pointPolygonTest(leftlane_coords, centroid, False) == 1:
            return "left"
        elif cv2.pointPolygonTest(rightlane_coords, centroid, False) == 1:
            return "right"
        else:
            return None

    return lane_detector


def init_direction_detector(camera_meta: dict) -> Callable:
    leftlane_ref = camera_meta["leftlane_ref"]
    rightlane_ref = camera_meta["rightlane_ref"]

    def direction_detector(lane, pt1, pt2):
        if lane == "left":
            return distance.euclidean(pt1, leftlane_ref) > distance.euclidean(pt2, leftlane_ref)
        else:
            return distance.euclidean(pt1, rightlane_ref) > distance.euclidean(pt2, rightlane_ref)

    return direction_detector
