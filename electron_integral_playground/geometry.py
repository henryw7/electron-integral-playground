
from dataclasses import dataclass

import numpy as np

@dataclass
class Molecule:
    elements: list[str]
    geometry: np.ndarray # Bohr
