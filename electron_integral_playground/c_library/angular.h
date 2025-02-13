
#pragma once

#define MAX_L 6
#define MAX_AUX_L 9

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define CARTESIAN_ORBITAL_COUNT(L) (((L) + 1) * ((L) + 2) / 2)

#define MAX_REGISTER_MATRIX_DOUBLE_COUNT (CARTESIAN_ORBITAL_COUNT(MAX_L) * CARTESIAN_ORBITAL_COUNT(MAX_AUX_L))

#define MAX_BOYS_ORDER MAX(MAX_L * 4 + 2, MAX_AUX_L * 2 + 2)
