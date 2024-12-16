
from dataclasses import dataclass

import numpy as np

@dataclass
class GaussianShell:
    angular: int
    spherical: bool
    i_ao_start: int
    i_atom: int
    primitive_exponents: np.ndarray
    primitive_coefficients: np.ndarray

angular_letter_to_l_value_map = { "S":0, "P":1, "D":2, "F":3, "G":4, "H":5, "I":6 }
def angular_letter_to_l_value(letter: str) -> int:
    assert len(letter) == 1
    letter = letter.upper()
    assert letter in angular_letter_to_l_value_map
    return angular_letter_to_l_value_map[letter]
