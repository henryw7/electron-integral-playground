
from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class Molecule:
    elements: List[str]
    geometry: np.ndarray # Bohr
