import numpy as np
from typing import Callable
from collections import deque, OrderedDict


class VehicleObject(object):

    def __init__(self, objid, centroid, rect, lane, direction, path, absent_count) -> None:
        self.objid = objid
        self.centroid = centroid
        self.rect = rect
        self.lane = lane
        self.direction = direction
        self.path = path
        self.absent_count = absent_count
        self.state = [0] * 4

        # state uncertainity covariance matrix
        self.P = np.matrix([[100., 0., 0., 0.],
                            [0., 100., 0., 0.],
                            [0., 0., 100., 0.],
                            [0., 0., 0., 100.]])


class BaseTracker(object):

    def __init__(
            self,
            lane_detector: Callable,
            direction_detector: Callable,
            maxdist: int,
            path_from_centroid: bool,
            max_absent: int,
            max_track_pts: int) -> None:
        self.lane_detector = lane_detector
        self.direction_detector = direction_detector
        self.maxdist = maxdist
        self.path_from_centroid = path_from_centroid
        self.max_absent = max_absent
        self.max_track_pts = max_track_pts
        self.next_objid = 0
        self.objects = OrderedDict()

    def _register_object(self, centroid, rect):
        self.next_objid += 1
        lane = self.lane_detector(centroid)
        if lane is None:
            self.next_objid -= 1
            return

        objid = lane + "_" + str(self.next_objid)
        self.objects[self.next_objid] = VehicleObject(objid, centroid, rect, lane, True, deque([]), 0)

        pt = centroid
        if not self.path_from_centroid:
            pt = (centroid[0], centroid[1]+(rect[3]-rect[1])//2)

        self.objects[self.next_objid].path.append(pt)

    def _deregister_object(self, obj_id) -> None:
        del self.objects[obj_id]

    def update(self, detections: list):
        raise NotImplementedError("Function `update` is implemented !")
