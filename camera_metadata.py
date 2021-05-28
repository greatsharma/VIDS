import numpy as np


CAMERA_METADATA = {
    "place5": {
        
        "lane1": {

            "lane_coords": np.array(
                    [(12, 408), (12, 452), (168, 465), (430, 100), (390, 97)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (56, 497),

            "mid_ref": [[(200, 241), (420, 259)], [(300, 171), (445, 183)], [(360, 119), (460, 127)]],

            "classupdate_line": [(12, 452), (348, 483), (178,297), (401,315)],

            "deregistering_line_rightdirection": [(395,93), (471,98)],
    
            "deregistering_line_wrongdirection": [(6,458), (345,494)],

            "angle": 0.86,
            
            "speed_reflines": [[(0,421), (500,421)], [(0,362), (500,362)], [(0,335), (500,335)], [(0,297), (500,297)], [(0,278), (500,278)]],

            "speedrefs_length": [6, 3, 6, 3], # in meters

            "speedinterval_length": 15,

        },

        "lane2": {

            "lane_coords": np.array(
                    [(168, 465), (348, 483), (469, 102), (430, 100)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (245, 510),

            "mid_ref": [[(200, 241), (420, 259)], [(300, 171), (445, 183)], [(360, 119), (460, 127)]],

            "classupdate_line": [(12, 452), (348, 483), (178,297), (401,315)],

            "deregistering_line_rightdirection": [(395,93), (471,98)],

            "deregistering_line_wrongdirection": [(6,458), (345,494)],

            "angle": 1.102,

            "speed_reflines": [[(0,421), (500,421)], [(0,362), (500,362)], [(0,335), (500,335)], [(0,297), (500,297)], [(0,278), (500,278)]],

            "speedrefs_length": [6, 3, 6, 3], # in meters

            "speedinterval_length": 15,

        },

        "lane3": {

            "lane_coords": np.array(
                    [(641, 477), (834,473), (574, 105), (530, 105)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (757, 523),

            "mid_ref": [[(535, 131), (662, 135)], [(550, 187), (733, 190)], [(575, 267), (825, 270)]],

            "classupdate_line": [(571, 241), (764, 242), (604, 354), (874, 345)],

            "deregistering_line_rightdirection": [(644, 488), (844, 484)],

            "deregistering_line_wrongdirection": [(529, 100), (613, 100)],

            "angle": -1.107,

            "speed_reflines": [[(550,266), (1000,266)], [(550,298), (1000,298)], [(550,319), (1000,319)], [(550,364), (1000,364)], [(550,398), (1000,398)]],

            "speedrefs_length": [6, 3, 6, 3], # in meters

            "speedinterval_length": 15,

        },

        "lane4": {

            "lane_coords": np.array(
                    [(801, 426), (950, 417), (945, 364), (630, 105), (574, 105)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (930, 506),

            "mid_ref": [[(535, 131), (662, 135)], [(550, 187), (733, 190)], [(575, 267), (825, 270)]],

            "classupdate_line": [(571, 241), (764, 242), (604, 354), (874, 345)],

            "deregistering_line_rightdirection": [(810,436), (957,424)],

            "deregistering_line_wrongdirection": [(529, 100), (613, 100)],

            "angle": -0.852,

            "speed_reflines": [[(550,266), (1000,266)], [(550,298), (1000,298)], [(550,319), (1000,319)], [(550,364), (1000,364)], [(550,398), (1000,398)]],

            "speedrefs_length": [6, 3, 6, 3], # in meters

            "speedinterval_length": 15,

        },

        "lane5": {

            "lane_coords": np.array(
                    [(12, 270), (12, 408), (366, 115), (300, 115)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (5, 350),

            "mid_ref": [[(200, 241), (420, 259)], [(300, 171), (445, 183)], [(360, 119), (460, 127)]],

            "classupdate_line": [(12, 270), (12, 408), (168,190), (224,231)],

            "deregistering_line_rightdirection": [(395,93), (471,98)],

            "deregistering_line_wrongdirection": [(6,458), (345,494)],

            "angle": 0.86,

        },

        "lane6": {

            "lane_coords": np.array(
                    [(945, 364), (935, 270), (720, 128), (658, 128)],
                    dtype=np.int32,
                ).reshape((-1, 1, 2)),

            "lane_ref": (954, 319),

            "mid_ref": [[(535, 131), (662, 135)], [(550, 187), (733, 190)], [(575, 267), (825, 270)]],

            "classupdate_line": [(768, 218), (800, 185), (945, 364), (935, 270)],

            "deregistering_line_rightdirection": [(810,436), (957,424)],

            "deregistering_line_wrongdirection": [(529, 100), (613, 100)],

            "angle": -0.852,

        },

        "initial_maxdistances": {
            "hmv": [35, 25, 18, 10],
            "lmv,auto,tw": [40, 30, 20, 10],
            "pedestrian,cattles": [18, 12, 10, 6],
        },

        "intesection_point_of_all_lanes": (503, -8),
    },
}


if __name__ == "__main__":
    from pprint import pprint
    pprint(CAMERA_METADATA)