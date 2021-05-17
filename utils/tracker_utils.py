from typing import Callable
from scipy.spatial import distance


def init_classupdate_line(camera_meta: dict) -> Callable:
    (A1x, A1y), (B1x, B1y) = camera_meta["lane1"]["classupdate_line"]
    (A2x, A2y), (B2x, B2y) = camera_meta["lane2"]["classupdate_line"]
    (A3x, A3y), (B3x, B3y) = camera_meta["lane3"]["classupdate_line"]
    (A4x, A4y), (B4x, B4y) = camera_meta["lane4"]["classupdate_line"]

    def within_interval(pt, lane):
        if lane == "1":
            return (pt[0] - A1x) * (B1y - A1y) - (pt[1] - A1y) * (B1x - A1x)
        elif lane == "2":
            return (pt[0] - A2x) * (B2y - A2y) - (pt[1] - A2y) * (B2x - A2x)
        elif lane == "3":
            return (pt[0] - A3x) * (B3y - A3y) - (pt[1] - A3y) * (B3x - A3x)
        else:
            return (pt[0] - A4x) * (B4y - A4y) - (pt[1] - A4y) * (B4x - A4x)

    return within_interval


def init_direction_detector(camera_meta: dict) -> Callable:
    lane1_ref = camera_meta["lane1"]["lane_ref"]
    lane2_ref = camera_meta["lane2"]["lane_ref"]
    lane3_ref = camera_meta["lane3"]["lane_ref"]
    lane4_ref = camera_meta["lane4"]["lane_ref"]

    def direction_detector(lane, pt1, pt2):
        if lane == "1":
            return distance.euclidean(pt1, lane1_ref) > distance.euclidean(
                pt2, lane1_ref
            )

        elif lane == "2":
            return distance.euclidean(pt1, lane2_ref) > distance.euclidean(
                pt2, lane2_ref
            )
        elif lane == "3":
            return distance.euclidean(pt1, lane3_ref) > distance.euclidean(
                pt2, lane3_ref
            )
        else:
            return distance.euclidean(pt1, lane4_ref) > distance.euclidean(
                pt2, lane4_ref
            )

    return direction_detector
