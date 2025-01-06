
#pragma once

#include "angular.h"

#define BOYS_SPLINE_N_POINTS 2961
const double boys_spline_x0_start = 0.2;
const double boys_spline_x_interval = 0.005;
const double one_over_boys_spline_x_interval = 1.0 / boys_spline_x_interval;

extern const double boys_cubic_spline_coefficient[MAX_L * 4 + 2 + 1][BOYS_SPLINE_N_POINTS * 4];
