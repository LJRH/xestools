from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class DataSet:
    # kind: "1D" or "2D"
    kind: str
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    # Optional 2D coordinates for curvilinear grids (used for energy transfer)
    x2d: Optional[np.ndarray] = None
    y2d: Optional[np.ndarray] = None
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    source: str = ""