import numpy as np


CAMERA_METADATA = {
    "1": {
        "cntrarea_thresh1": 400,
        "cntrarea_thresh2": 300,
        "leftlane_coords": np.array([(180, 530), (940, 530), (940, 300), (250, 40), (10, 40)], dtype=np.int32).reshape((-1, 1, 2)),
        "rightlane_coords": np.array([(950, 280), (950, 110), (420, 10), (250, 30)], dtype=np.int32).reshape((-1, 1, 2)),
        "leftlane_ref": (540, 530),
        "rightlane_ref": (335, 20),
        "mid_ref": (530, 160),
        "max_distance": 20,
    },

    "2": {
        "cntrarea_thresh1": 400,
        "cntrarea_thresh2": 300,
        "leftlane_coords": np.array([(150, 530), (940, 530), (940, 350), (5, 120), (5, 300)], dtype=np.int32).reshape((-1, 1, 2)),
        "rightlane_coords": np.array([(940, 310), (940, 200), (150, 80), (100, 130)], dtype=np.int32).reshape((-1, 1, 2)),
        "leftlane_ref": (940, 460),
        "rightlane_ref": (125, 105),
        "mid_ref": (520, 250),
        "max_distance": 20,
    },

    "3": {
        "cntrarea_thresh1": 300,
        "cntrarea_thresh2": 150,
        "leftlane_coords": np.array([(360, 350), (640, 350), (380, 150), (335, 150)], dtype=np.int32).reshape((-1, 1, 2)),
        "rightlane_coords": np.array([(5, 315), (260, 350), (315, 150), (270, 150)], dtype=np.int32).reshape((-1, 1, 2)),
        "leftlane_ref": (500, 350),
        "rightlane_ref": (292, 150),
        "mid_ref": (320, 200),
        "max_distance": 20,
    },

    "4": {
        "cntrarea_thresh1": 400,
        "cntrarea_thresh2": 200,
        "leftlane_coords": np.array([(120, 400), (560, 400), (615, 90), (540, 90)], dtype=np.int32).reshape((-1, 1, 2)),
        "rightlane_coords": np.array([(640, 400), (940, 400), (960, 300), (680, 90), (620, 90)], dtype=np.int32).reshape((-1, 1, 2)),
        "leftlane_ref": (340, 400),
        "rightlane_ref": (650, 90),
        "mid_ref": (615, 175),
        "max_distance": 20,
    },
}
