from typing import Callable
from scipy.spatial import distance


def init_classupdate_line(camera_meta: dict) -> Callable:
    (A1x, A1y), (B1x, B1y), (C1x, C1y), (D1x, D1y) = camera_meta["lane1"]["classupdate_line"]
    (A2x, A2y), (B2x, B2y), (C2x, C2y), (D2x, D2y) = camera_meta["lane2"]["classupdate_line"]
    (A3x, A3y), (B3x, B3y), (C3x, C3y), (D3x, D3y) = camera_meta["lane3"]["classupdate_line"]
    (A4x, A4y), (B4x, B4y), (C4x, C4y), (D4x, D4y) = camera_meta["lane4"]["classupdate_line"]
    (A5x, A5y), (B5x, B5y), (C5x, C5y), (D5x, D5y) = camera_meta["lane5"]["classupdate_line"]
    (A6x, A6y), (B6x, B6y), (C6x, C6y), (D6x, D6y) = camera_meta["lane6"]["classupdate_line"]

    def within_interval(pt, lane):
        Px, Py = pt

        if lane == "1":
            position1 = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)
            position2 = (Px - C1x) * (D1y - C1y) - (Py - C1y) * (D1x - C1x)
            return position1 > 0 and position2 < 0

        elif lane == "2":
            position1 = (Px - A2x) * (B2y - A2y) - (Py - A2y) * (B2x - A2x)
            position2 = (Px - C2x) * (D2y - C2y) - (Py - C2y) * (D2x - C2x)
            return position1 > 0 and position2 < 0

        elif lane == "3":
            position1 = (Px - A3x) * (B3y - A3y) - (Py - A3y) * (B3x - A3x)
            position2 = (Px - C3x) * (D3y - C3y) - (Py - C3y) * (D3x - C3x)
            return position1 < 0 and position2 > 0

        elif lane == "4":
            position1 = (Px - A4x) * (B4y - A4y) - (Py - A4y) * (B4x - A4x)
            position2 = (Px - C4x) * (D4y - C4y) - (Py - C4y) * (D4x - C4x)
            return position1 < 0 and position2 > 0

        elif lane == "5":
            position1 = (Px - A5x) * (B5y - A5y) - (Py - A5y) * (B5x - A5x)
            position2 = (Px - C5x) * (D5y - C5y) - (Py - C5y) * (D5x - C5x)
            return position1 > 0 and position2 < 0

        elif lane == "6":
            position1 = (Px - A6x) * (B6y - A6y) - (Py - A6y) * (B6x - A6x)
            position2 = (Px - C6x) * (D6y - C6y) - (Py - C6y) * (D6x - C6x)
            return position1 < 0 and position2 > 0

    return within_interval


def init_direction_detector(camera_meta: dict) -> Callable:
    lane1_ref = camera_meta["lane1"]["lane_ref"]
    lane2_ref = camera_meta["lane2"]["lane_ref"]
    lane3_ref = camera_meta["lane3"]["lane_ref"]
    lane4_ref = camera_meta["lane4"]["lane_ref"]
    lane5_ref = camera_meta["lane5"]["lane_ref"]
    lane6_ref = camera_meta["lane6"]["lane_ref"]

    def direction_detector(lane, pt1, pt2):
        if lane == "1":
            return distance.euclidean(pt1, lane1_ref) - distance.euclidean(
                pt2, lane1_ref
            )
        elif lane == "2":
            return distance.euclidean(pt1, lane2_ref) - distance.euclidean(
                pt2, lane2_ref
            )
        elif lane == "3":
            return distance.euclidean(pt2, lane3_ref) - distance.euclidean(
                pt1, lane3_ref
            )
        elif lane == "4":
            return distance.euclidean(pt2, lane4_ref) - distance.euclidean(
                pt1, lane4_ref
            )
        elif lane == "5":
            return distance.euclidean(pt1, lane5_ref) - distance.euclidean(
                pt2, lane5_ref
            )
        elif lane == "6":
            return distance.euclidean(pt2, lane6_ref) - distance.euclidean(
                pt1, lane6_ref
            )

    return direction_detector


def init_speed_detector(camera_meta: dict) -> Callable:
    lane1_ref1, lane1_ref2 = camera_meta["lane1"]["speed_reflines"]
    lane2_ref1, lane2_ref2 = camera_meta["lane2"]["speed_reflines"]
    lane3_ref1, lane3_ref2 = camera_meta["lane3"]["speed_reflines"]
    lane4_ref1, lane4_ref2 = camera_meta["lane4"]["speed_reflines"]

    lane_refs = {
        "lane1": {
            "ref1": lane1_ref1,
            "ref2": lane1_ref2,
        },
        "lane2": {
            "ref1": lane2_ref1,
            "ref2": lane2_ref2,
        },
        "lane3": {
            "ref1": lane3_ref1,
            "ref2": lane3_ref2,
        },
        "lane4": {
            "ref1": lane4_ref1,
            "ref2": lane4_ref2,
        },
    }

    # meterdistance_between_interval
    interval_dist = {
        "lane1": 12,
        "lane2": 12,
        "lane3": 15,
        "lane4": 15,
    }

    def speed_detector(obj):        
        l = len(obj.framecount_speedrefs)
        
        Px,Py = obj.path[-1]
        
        if l == 0:
            (A1x, A1y), (B1x, B1y) = lane_refs[f"lane{obj.lane}"]["ref2"]
            position = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)

            if (obj.lane in ["1", "2"] and position > 0) or (obj.lane in ["3", "4"] and position < 0):
                obj.speed = None
                obj.framecount_speedrefs.extend([None,None])
                return

        (A1x, A1y), (B1x, B1y) = lane_refs[f"lane{obj.lane}"][f"ref{l+1}"]
        position = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)

        if (obj.lane in ["1", "2"] and position > 0) or (obj.lane in ["3", "4"] and position < 0):
            obj.framecount_speedrefs.append(len(obj.path))
        
        if not obj.speed and len(obj.framecount_speedrefs)==2:
            speed = interval_dist[f"lane{obj.lane}"] / (obj.framecount_speedrefs[1] - obj.framecount_speedrefs[0]) # speed in meter/frame
            speed /= (0.0625) # speed in meter/seconds, 1/16 = 0.0625, 16 is input-fps
            speed *= 3.6 # speed in kmph
            obj.speed = int(speed)

    return speed_detector