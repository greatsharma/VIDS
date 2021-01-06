import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from trackers import BaseTracker


class CentroidTracker(BaseTracker):

    def update(self, detections: list):
        if len(detections) == 0:
            for obj_id, obj in self.objects.items():
                obj.absent_count += 1

                if obj.absent_count > self.max_absent:
                    self._deregister_object(obj_id)

            return self.objects

        detected_centroids = np.zeros((len(detections), 2), dtype="int")

        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            detected_centroids[i] = (x1+x2)//2, (y1+y2)//2

        if len(self.objects) == 0:
            for ctr, rect in zip(detected_centroids, detections):
                self._register_object(tuple(ctr), rect)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = [self.objects[obj_id].centroid for obj_id in obj_ids]

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
                    self.objects[obj_id].centroid = tuple(detected_centroids[col])
                    self.objects[obj_id].rect = detections[col]

                    if len(self.objects[obj_id].path) > 5:
                        self.objects[obj_id].direction = self.direction_detector(
                            self.objects[obj_id].lane,
                            self.objects[obj_id].path[-1],
                            self.objects[obj_id].path[-5]
                        )

                    new_pt = (
                        detected_centroids[col]
                        if self.path_from_centroid else
                        (detected_centroids[col][0], detected_centroids[col][1] + (detections[col][3]-detections[col][1])//2)
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

                    if self.objects[obj_id].absent_count > self.max_absent:
                        self._deregister_object(obj_id)
            else:
                for col in unused_cols:
                    self._register_object(tuple(detected_centroids[col]), detections[col])

        return self.objects
