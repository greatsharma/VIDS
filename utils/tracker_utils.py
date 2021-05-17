from typing import Callable
from scipy.spatial import distance


def init_classupdate_line(camera_meta: dict) -> Callable:
    (Ax, Ay), (Bx, By) = camera_meta["classupdate_line"]

    def within_interval(pt):
        return (pt[0] - Ax) * (By - Ay) - (pt[1] - Ay) * (Bx - Ax)

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
