
#pragma once

#include "angular.h"

#define BOYS_CHEBYSHEV_POLYNOMIAL_ORDER 10
#define BOYS_CHEBYSHEV_N_INTERVAL 18
constexpr double boys_chebyshev_x0 = 0.0;
constexpr double boys_chebyshev_x_interval = 1.0;
constexpr double one_over_boys_chebyshev_x_interval = 1.0 / boys_chebyshev_x_interval;

extern const double boys_chebyshev_coefficient[MAX_BOYS_ORDER + 1][BOYS_CHEBYSHEV_N_INTERVAL][BOYS_CHEBYSHEV_POLYNOMIAL_ORDER + 1];
