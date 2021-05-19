import numpy as np


CAMERA_METADATA = {
    "place5": {
        
        "lane1": {

            "lane_coords": np.array(
                    [(12, 452), (168, 465), (430, 100), (394, 97)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (56, 497),

            "mid_ref": [(262,218), (427,232)],

            "countinterval": [(86,380), (372,403), (184,289), (404,304)],

            "classupdate_line": [(12, 452), (348, 483), (178,297), (401,315)],

            "deregistering_line_rightdirection": [(395,93), (471,98)],
    
            "deregistering_line_wrongdirection": [(6,458), (345,494)],

            "angle": 0.86,

        },

        "lane2": {

            "lane_coords": np.array(
                    [(168, 465), (348, 483), (469, 102), (430, 100)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (245, 510),

            "mid_ref": [(262,218), (427,232)],

            "countinterval": [(86,380), (372,403), (184,289), (404,304)],

            "classupdate_line": [(12, 452), (348, 483), (178,297), (401,315)],

            "deregistering_line_rightdirection": [(395,93), (471,98)],

            "deregistering_line_wrongdirection": [(6,458), (345,494)],

            "angle": 1.102,

        },

        "lane3": {

            "lane_coords": np.array(
                    [(641, 477), (834,473), (574, 105), (530, 105)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (757, 523),

            "mid_ref": [(565, 220), (745, 220)],

            "countinterval": [(577,267), (788,265), (608,366), (887,359)],

            "classupdate_line": [(571, 241), (764, 242), (604, 354), (874, 345)],

            "deregistering_line_rightdirection": [(644, 488), (844, 484)],

            "deregistering_line_wrongdirection": [(529, 100), (613, 100)],

            "angle": -1.107,

        },

        "lane4": {

            "lane_coords": np.array(
                    [(801, 426), (950, 417), (615, 105), (574, 105)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (930, 506),

            "mid_ref": [(565, 220), (745, 220)],

            "countinterval": [(577,267), (788,265), (608,366), (887,359)],

            "classupdate_line": [(571, 241), (764, 242), (604, 354), (874, 345)],

            "deregistering_line_rightdirection": [(810,436), (957,424)],

            "deregistering_line_wrongdirection": [(529, 100), (613, 100)],

            "angle": -0.852,

        },

        "initial_maxdistances": {
            "hmv": 35, #[15,30], # [above_midref, below_midref]
            "lmv,auto,tw": 45, #[25,50],
            "pedestrian,cattles": 12, #[6,12],
        },
    },
}


if __name__ == "__main__":
    from pprint import pprint
    pprint(CAMERA_METADATA)
