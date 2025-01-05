
#pragma once

#include <math.h>

#include "angular.h"

/*
    We've nomalized each shell according to the normalization of pure $x$ component (px, dx^2, fx^3, etc.)
    For other components, the additional normalization of the following form will apply:
    $$\sqrt{ \frac{i_x! i_y! i_z!}{(2i_x)! (2i_y)! (2i_z)!} \frac{(2L)!}{L!} }$$
*/

static const double kernel_cartesian_normalization_constants[MAX_L + 1][(MAX_L + 1) * (MAX_L + 2) / 2]
{
    { 1.0, },
    { 1.0,             1.0,             1.0, },
    { 1.0,       sqrt(3.0),       sqrt(3.0),             1.0,       sqrt(3.0),             1.0, },
    { 1.0,       sqrt(5.0),       sqrt(5.0),       sqrt(5.0),      sqrt(15.0),       sqrt(5.0),             1.0,       sqrt(5.0),       sqrt(5.0),             1.0, },
    { 1.0,       sqrt(7.0),       sqrt(7.0),  sqrt(35.0/3.0),      sqrt(35.0),  sqrt(35.0/3.0),       sqrt(7.0),      sqrt(35.0),      sqrt(35.0),       sqrt(7.0),             1.0,       sqrt(7.0),  sqrt(35.0/3.0),       sqrt(7.0),             1.0, },
    { 1.0,       sqrt(9.0),       sqrt(9.0),      sqrt(21.0),      sqrt(63.0),      sqrt(21.0),      sqrt(21.0),     sqrt(105.0),     sqrt(105.0),      sqrt(21.0),       sqrt(9.0),      sqrt(63.0),     sqrt(105.0),      sqrt(63.0),       sqrt(9.0),             1.0,       sqrt(9.0),      sqrt(21.0),      sqrt(21.0),       sqrt(9.0),             1.0, },
    { 1.0,      sqrt(11.0),      sqrt(11.0),      sqrt(33.0),      sqrt(99.0),      sqrt(33.0), sqrt(231.0/5.0),     sqrt(231.0),     sqrt(231.0), sqrt(231.0/5.0),      sqrt(33.0),     sqrt(231.0),     sqrt(385.0),     sqrt(231.0),      sqrt(33.0),      sqrt(11.0),      sqrt(99.0),     sqrt(231.0),     sqrt(231.0),      sqrt(99.0),      sqrt(11.0),             1.0,      sqrt(11.0),      sqrt(33.0), sqrt(231.0/5.0),      sqrt(33.0),      sqrt(11.0),             1.0,},
};

template<int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_L && j_L >= 0 && j_L <= MAX_L)
static void kernel_cartesian_normalize(double S_cartesian[(i_L + 1) * (i_L + 2) / 2 * (j_L + 1) * (j_L + 2) / 2])
{
    if constexpr (i_L <= 1 && j_L <= 1)
        return;

    constexpr int n_density_i = (i_L + 1) * (i_L + 2) / 2;
    constexpr int n_density_j = (j_L + 1) * (j_L + 2) / 2;
#pragma unroll
    for (int i = 0; i < n_density_i; i++)
    {
#pragma unroll
        for (int j = 0; j < n_density_j; j++)
        {
            S_cartesian[i * n_density_j + j] *= kernel_cartesian_normalization_constants[i_L][i] * kernel_cartesian_normalization_constants[j_L][j];
        }
    }
}
