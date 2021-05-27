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


def init_position_wrt_midrefs(camera_meta: dict) -> Callable:
    lane_refs = {f"{l}": {f"ref{r}": camera_meta[f"lane{l}"]["mid_ref"][r-1] for r in [1,2,3]} for l in [1,2,3,4,5,6]}

    def pos_wrt_midrefs__detector(lane, pt):

        (Ax, Ay), (Bx, By) = lane_refs[lane]["ref1"]
        position_wrt_midref = (pt[0] - Ax) * (By - Ay) - (pt[1] - Ay) * (Bx - Ax)

        if position_wrt_midref < 0:
            return 0

        (Ax, Ay), (Bx, By) = lane_refs[lane]["ref2"]
        position_wrt_midref = (pt[0] - Ax) * (By - Ay) - (pt[1] - Ay) * (Bx - Ax)
        
        if position_wrt_midref < 0:
            return 1

        (Ax, Ay), (Bx, By) = lane_refs[lane]["ref3"]
        position_wrt_midref = (pt[0] - Ax) * (By - Ay) - (pt[1] - Ay) * (Bx - Ax)
        
        if position_wrt_midref < 0:
            return 2

        return 3

    return pos_wrt_midrefs__detector


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


def _interval_wrt_speedrefs(Px, Py, lane, lane_refs):
    "we are returning interval"

    (A1x, A1y), (B1x, B1y) = lane_refs[lane][5]
    position = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)
    
    if (lane in ["1", "2"] and position > 0) or (lane in ["3", "4"] and position < 0):
        return 5

    (A1x, A1y), (B1x, B1y) = lane_refs[lane][4]
    position = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)
    
    if (lane in ["1", "2"] and position > 0) or (lane in ["3", "4"] and position < 0):
        return 4

    (A1x, A1y), (B1x, B1y) = lane_refs[lane][3]
    position = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)
    
    if (lane in ["1", "2"] and position > 0) or (lane in ["3", "4"] and position < 0):
        return 3

    (A1x, A1y), (B1x, B1y) = lane_refs[lane][2]
    position = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)
    
    if (lane in ["1", "2"] and position > 0) or (lane in ["3", "4"] and position < 0):
        return 2

    (A1x, A1y), (B1x, B1y) = lane_refs[lane][1]
    position = (Px - A1x) * (B1y - A1y) - (Py - A1y) * (B1x - A1x)
    
    if (lane in ["1", "2"] and position > 0) or (lane in ["3", "4"] and position < 0):
        return 1

    return 0


def _line_intersect(A1, A2, B1, B2):
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
 
    return int(x), int(y)


def _project_point_on_line(lane, point_to_project, line_coords, point_along_which_to_project, direction_of_projection):
    pt1 = point_to_project
    pt2 = point_along_which_to_project

    if (direction_of_projection == "lower" and lane in ["1", "2"]) or (direction_of_projection == "upper" and lane in ["3", "4"]):
        ratio = 3.8
    elif (direction_of_projection == "upper" and lane in ["1", "2"]) or (direction_of_projection == "lower" and lane in ["3", "4"]):
        ratio = -3.8

    cx = int(pt2[0] + (pt1[0]-pt2[0]) * ratio)
    cy = int(pt2[1] + (pt1[1]-pt2[1]) * ratio)
    pt3 = (cx, cy)

    return _line_intersect(pt1, pt3, line_coords[0], line_coords[1])


def init_avgspeed_detector(camera_meta: dict) -> Callable:
    lane_refs = {f"{l}": {r: camera_meta[f"lane{l}"]["speed_reflines"][r-1] for r in [1,2,3,4,5]} for l in [1,2,3,4]}

    intesection_point_of_all_lanes = camera_meta["intesection_point_of_all_lanes"]

    speedinterval_length = {f"{l}": camera_meta[f"lane{l}"]["speedinterval_length"] for l in [1,2,3,4]}

    speedrefs_length = {
        f"{l}": {i: camera_meta[f"lane{l}"]["speedrefs_length"][i-1] for i in [1,2,3,4]} for l in [1,2,3,4]
    }

    def avgspeed_detector(obj, frame_count):        
        l = len(obj.avgspeed_metadata)
        Px,Py = (obj.state[0], obj.state[2])
        interval = _interval_wrt_speedrefs(Px, Py, obj.lane, lane_refs)

        if l == 0:
            if interval in [4, 5]:
                obj.avgspeed = None
            elif interval in [1, 2, 3]:
                projected_pt1 = _project_point_on_line(obj.lane, (Px,Py), lane_refs[obj.lane][interval], intesection_point_of_all_lanes, direction_of_projection="lower")
                projected_pt2 = _project_point_on_line(obj.lane, (Px,Py), lane_refs[obj.lane][interval+1], intesection_point_of_all_lanes, direction_of_projection="upper")

                pixle_distance1 = distance.euclidean((Px,Py), projected_pt1)
                pixle_distance2 = distance.euclidean((Px,Py), projected_pt2)
                pixles_per_meter = (pixle_distance1 + pixle_distance2) / speedrefs_length[obj.lane][interval]
                distance_covered_in_metres = - pixle_distance1 / pixles_per_meter
                
                if interval > 1:
                    distance_covered_in_metres -=  speedrefs_length[obj.lane][1]

                if interval > 2:
                    distance_covered_in_metres -=  speedrefs_length[obj.lane][2]

                obj.avgspeed_metadata[frame_count] = distance_covered_in_metres

            return

        if interval == 5:
            obj.avgspeed = None

        if interval == 4:
            projected_pt1 = _project_point_on_line(obj.lane, (Px,Py), lane_refs[obj.lane][4], intesection_point_of_all_lanes, direction_of_projection="lower")
            projected_pt2 = _project_point_on_line(obj.lane, (Px,Py), lane_refs[obj.lane][5], intesection_point_of_all_lanes, direction_of_projection="upper")

            pixle_distance1 = distance.euclidean((Px,Py), projected_pt1)
            pixle_distance2 = distance.euclidean((Px,Py), projected_pt2)
            pixles_per_meter = (pixle_distance1 + pixle_distance2) / speedrefs_length[obj.lane][4]
            distance_covered_in_metres = pixle_distance1 / pixles_per_meter

            obj.avgspeed_metadata[frame_count] = distance_covered_in_metres

            distance_covered_in_metres = speedinterval_length[obj.lane] + sum(obj.avgspeed_metadata.values())
            frame_counts = list(obj.avgspeed_metadata.keys())
            frame_diff = frame_counts[1] - frame_counts[0]

            speed = distance_covered_in_metres / frame_diff # speed in meter/frame
            speed /= (0.0625) # speed in meter/seconds, 1/16 = 0.0625, 16 is input-fps
            speed *= 3.6 # speed in kmph
            obj.avgspeed = round(speed, 1)

    return avgspeed_detector


def init_instspeed_detector(camera_meta: dict) -> Callable:
    lane_refs = {f"{l}": {r: camera_meta[f"lane{l}"]["speed_reflines"][r-1] for r in [1,2,3,4,5]} for l in [1,2,3,4]}

    intesection_point_of_all_lanes = camera_meta["intesection_point_of_all_lanes"]

    speedrefs_length = {
        f"{l}": {i: camera_meta[f"lane{l}"]["speedrefs_length"][i-1] for i in [1,2,3,4]} for l in [1,2,3,4]
    }

    def instspeed_detector(obj, curr_framecount):
        from pprint import pprint

        Px,Py = (obj.state[0], obj.state[2])
        interval = _interval_wrt_speedrefs(Px, Py, obj.lane, lane_refs)

        to_log = 3

        if interval == 0:
            return

        if interval == 5:
            obj.instspeed_list.append(None)
            return

        if not len(obj.instspeed_metadata):
            if interval == 4:
                obj.instspeed_list.append(None)
                return

        obj.instspeed_metadata[curr_framecount] = (interval, (Px,Py))
        if obj.objid == to_log:
            print(f"\n\n")
            pprint(obj.instspeed_metadata)

        first_framecount = next(iter(obj.instspeed_metadata))

        if curr_framecount >= first_framecount + 5:

            distance_covered_in_metres = 0
            prev_interval, prev_position =  obj.instspeed_metadata[first_framecount]
            curr_interval, curr_position =  obj.instspeed_metadata[curr_framecount]

            if prev_interval == curr_interval:

                projected_pt1 = _project_point_on_line(obj.lane, prev_position, lane_refs[obj.lane][prev_interval], intesection_point_of_all_lanes, direction_of_projection="lower")
                projected_pt2 = _project_point_on_line(obj.lane, prev_position, lane_refs[obj.lane][prev_interval+1], intesection_point_of_all_lanes, direction_of_projection="upper")

                pixle_distance1 = distance.euclidean(prev_position, projected_pt1)
                pixle_distance2 = distance.euclidean(prev_position, projected_pt2)
                pixles_per_meter = (pixle_distance1 + pixle_distance2) / speedrefs_length[obj.lane][prev_interval]
                distance_covered_in_metres -= pixle_distance1 / pixles_per_meter

                if obj.objid == to_log:
                    print(pixle_distance1 / pixles_per_meter, end=", ")

                projected_pt1 = _project_point_on_line(obj.lane, curr_position, lane_refs[obj.lane][curr_interval], intesection_point_of_all_lanes, direction_of_projection="lower")
                projected_pt2 = _project_point_on_line(obj.lane, curr_position, lane_refs[obj.lane][curr_interval+1], intesection_point_of_all_lanes, direction_of_projection="upper")

                pixle_distance1 = distance.euclidean(curr_position, projected_pt1)
                pixle_distance2 = distance.euclidean(curr_position, projected_pt2)
                pixles_per_meter = (pixle_distance1 + pixle_distance2) / speedrefs_length[obj.lane][curr_interval]
                distance_covered_in_metres += pixle_distance1 / pixles_per_meter

                if obj.objid == to_log:
                    print(pixle_distance1 / pixles_per_meter, end=", ")

            else:

                projected_pt1 = _project_point_on_line(obj.lane, prev_position, lane_refs[obj.lane][prev_interval], intesection_point_of_all_lanes, direction_of_projection="lower")
                projected_pt2 = _project_point_on_line(obj.lane, prev_position, lane_refs[obj.lane][prev_interval+1], intesection_point_of_all_lanes, direction_of_projection="upper")

                pixle_distance1 = distance.euclidean(prev_position, projected_pt1)
                pixle_distance2 = distance.euclidean(prev_position, projected_pt2)
                pixles_per_meter = (pixle_distance1 + pixle_distance2) / speedrefs_length[obj.lane][prev_interval]
                distance_covered_in_metres += pixle_distance2 / pixles_per_meter

                if obj.objid == to_log:
                    print(pixle_distance2 / pixles_per_meter, end=", ")

                projected_pt1 = _project_point_on_line(obj.lane, curr_position, lane_refs[obj.lane][curr_interval], intesection_point_of_all_lanes, direction_of_projection="lower")
                projected_pt2 = _project_point_on_line(obj.lane, curr_position, lane_refs[obj.lane][curr_interval+1], intesection_point_of_all_lanes, direction_of_projection="upper")

                pixle_distance1 = distance.euclidean(curr_position, projected_pt1)
                pixle_distance2 = distance.euclidean(curr_position, projected_pt2)
                pixles_per_meter = (pixle_distance1 + pixle_distance2) / speedrefs_length[obj.lane][curr_interval]
                distance_covered_in_metres += pixle_distance1 / pixles_per_meter

                if obj.objid == to_log:
                    print(pixle_distance1 / pixles_per_meter, end=", ")

                if curr_interval - prev_interval >= 2:
                    distance_covered_in_metres +=  speedrefs_length[obj.lane][curr_interval-1]

                if curr_interval - prev_interval >= 3:
                    distance_covered_in_metres +=  speedrefs_length[obj.lane][curr_interval-2]

            speed = distance_covered_in_metres / (curr_framecount - first_framecount) # speed in meter/frame
            speed /= (0.0625) # speed in meter/seconds, 1/16 = 0.0625, 16 is input-fps
            speed *= 3.6 # speed in kmph
            obj.instspeed_list.append(round(speed, 1))

            if obj.objid == to_log:
                print(distance_covered_in_metres, (curr_framecount - first_framecount), obj.instspeed_list)

    return instspeed_detector
