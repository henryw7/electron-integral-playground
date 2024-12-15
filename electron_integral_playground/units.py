
from enum import Enum

class LengthUnits(Enum):
    BOHR = 1
    ANGSTROM = 2

bohr_to_angstrom = 5.2917721054482e-1
angstrom_to_bohr = 1.0 / bohr_to_angstrom
