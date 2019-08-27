from enum import Enum


class ModelTuningType(Enum):
    GRID_SEARCH = 0
    EVALUATE_BEST = 1