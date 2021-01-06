import numpy as np
from scipy.spatial import distance
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment

from trackers import BaseTracker

# time interval
dt = 1.

# change in state due external forces
u = np.zeros((4, 1))

I = np.identity(4)

# state transition matrix assuming constant velocity model
# new_pos = old_pos + velocity * dt
F = np.matrix([[1., 0., dt, 0.],
               [0., 1., 0., dt],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])

# measurement function (maps the states to measurements)
H = np.matrix([[1., 0., 0., 0.],
               [0., 1., 0., 0.]])

# measurement noise covariance (measurement is done only for position not velocity)
R = np.matrix([[0.01, 0.],
               [0., 0.01]])

# Q is process covariance
Q_comp_mat = np.array([[dt**4/2., dt**3/2.],
                       [dt**3/2., dt**2]])
Q = block_diag(Q_comp_mat, Q_comp_mat)


class KalmanTracker(BaseTracker):

    def _apply_kf(self, obj_id, z):
        global dt, u, I, F, H, R, Q

        x = self.objects[obj_id].state
        x = np.array([x]).T

        # prediction steps
        x = F * x + u  # updating state, x' = x + v.dt + u
        self.objects[obj_id].P = F * self.objects[obj_id].P * F.T + Q

        if not z is None:
            P_ = self.objects[obj_id].P
            # measurement steps
            z = np.array(z)
            z = z.reshape(1, 2)
            e = z.T - H * x
            S = H * P_ * H.T + R
            K = P_ * H.T * np.linalg.inv(S)
            x += K*e
            self.objects[obj_id].P = (I - K * H) * P_

        x = x.astype(int).ravel().tolist()
        self.objects[obj_id].state = tuple(x[0])

    def update(self, detections: list):
        if len(detections) == 0:
            for obj_id, obj in self.objects.items():
                obj.absent_count += 1
                self._apply_kf(obj_id, None)

                if obj.absent_count > self.max_absent:
                    self._deregister_object(obj_id)

            return self.objects

        detected_centroids = np.zeros((len(detections), 2), dtype="int")

        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            detected_centroids[i] = (x1+x2)//2, (y1+y2)//2

        if len(self.objects) == 0:
            for ctr, rect in zip(detected_centroids, detections):
                i_count = self.next_objid
                self._register_object(tuple(ctr), rect)
                if self.next_objid > i_count:
                    self._apply_kf(self.next_objid, ctr)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = [self.objects[obj_id].state[:2] for obj_id in obj_ids]

            D = distance.cdist(np.array(obj_centroids), detected_centroids)

            rows, cols = linear_sum_assignment(D)
            rows, cols = rows.tolist(), cols.tolist()

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row][col] <= self.maxdist:
                    obj_id = obj_ids[row]
                    self._apply_kf(obj_id, detected_centroids[col])
                    self.objects[obj_id].rect = detections[col]

                    if len(self.objects[obj_id].path) > 15:
                        self.objects[obj_id].direction = self.direction_detector(
                            self.objects[obj_id].lane,
                            self.objects[obj_id].path[-1],
                            self.objects[obj_id].path[-15]
                        )

                    new_pt = (
                        self.objects[obj_id].state[:2]
                        if self.path_from_centroid else
                        (self.objects[obj_id].state[0], self.objects[obj_id].state[1] + (detections[col][3]-detections[col][1])//2)
                    )

                    if len(self.objects[obj_id].path) < self.max_track_pts:
                        self.objects[obj_id].path.append(new_pt)
                    else:
                        self.objects[obj_id].path.rotate(-1)
                        self.objects[obj_id].path[self.max_track_pts-1] = new_pt

                    self.objects[obj_id].absent_count = 0

                    used_rows.add(row)
                    used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    obj_id = obj_ids[row]
                    self.objects[obj_id].absent_count += 1
                    self._apply_kf(obj_id, None)

                    if self.objects[obj_id].absent_count > self.max_absent:
                        self._deregister_object(obj_id)
            else:
                for col in unused_cols:
                    i_count = self.next_objid
                    self._register_object(tuple(detected_centroids[col]), detections[col])
                    if self.next_objid > i_count:
                        self._apply_kf(self.next_objid, detected_centroids[col])

        return self.objects
