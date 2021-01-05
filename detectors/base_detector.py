import numpy as np
from typing import Callable


class BaseDetector(object):

    def __init__(self, is_valid_cntrarea: Callable, sub_type: str) -> None:
        self.is_valid_cntrarea = is_valid_cntrarea
        self.sub_type = sub_type

    def detect(self, curr_frame: np.ndarray) -> list:
        raise NotImplementedError("Function `detect` is not implemented !")
