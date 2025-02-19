
#pragma once

#include <math.h>

#include "angular.h"

static const double kernel_cartesian_normalization_constants[MAX_AUX_L + 1][CARTESIAN_ORBITAL_COUNT(MAX_AUX_L)]
{
    { 1.0, },
    { 1.0,                1.0,                1.0, },
    { 1.0,          sqrt(3.0),          sqrt(3.0),                1.0,          sqrt(3.0),                1.0, },
    { 1.0,          sqrt(5.0),          sqrt(5.0),          sqrt(5.0),         sqrt(15.0),          sqrt(5.0),                1.0,          sqrt(5.0),          sqrt(5.0),                1.0, },
    { 1.0,          sqrt(7.0),          sqrt(7.0),     sqrt(35.0/3.0),         sqrt(35.0),     sqrt(35.0/3.0),          sqrt(7.0),         sqrt(35.0),         sqrt(35.0),          sqrt(7.0),                1.0,          sqrt(7.0),     sqrt(35.0/3.0),          sqrt(7.0),                1.0, },
    { 1.0,          sqrt(9.0),          sqrt(9.0),         sqrt(21.0),         sqrt(63.0),         sqrt(21.0),         sqrt(21.0),        sqrt(105.0),        sqrt(105.0),         sqrt(21.0),          sqrt(9.0),         sqrt(63.0),        sqrt(105.0),         sqrt(63.0),          sqrt(9.0),                1.0,          sqrt(9.0),         sqrt(21.0),         sqrt(21.0),          sqrt(9.0),                1.0, },
    { 1.0,         sqrt(11.0),         sqrt(11.0),         sqrt(33.0),         sqrt(99.0),         sqrt(33.0),    sqrt(231.0/5.0),        sqrt(231.0),        sqrt(231.0),    sqrt(231.0/5.0),         sqrt(33.0),        sqrt(231.0),        sqrt(385.0),        sqrt(231.0),         sqrt(33.0),         sqrt(11.0),         sqrt(99.0),        sqrt(231.0),        sqrt(231.0),         sqrt(99.0),         sqrt(11.0),                1.0,         sqrt(11.0),         sqrt(33.0),    sqrt(231.0/5.0),         sqrt(33.0),         sqrt(11.0),                1.0, },
    { 1.0,         sqrt(13.0),         sqrt(13.0),    sqrt(143.0/3.0),        sqrt(143.0),    sqrt(143.0/3.0),    sqrt(429.0/5.0),        sqrt(429.0),        sqrt(429.0),    sqrt(429.0/5.0),    sqrt(429.0/5.0),   sqrt(3003.0/5.0),       sqrt(1001.0),   sqrt(3003.0/5.0),    sqrt(429.0/5.0),    sqrt(143.0/3.0),        sqrt(429.0),       sqrt(1001.0),       sqrt(1001.0),        sqrt(429.0),    sqrt(143.0/3.0),         sqrt(13.0),        sqrt(143.0),        sqrt(429.0),   sqrt(3003.0/5.0),        sqrt(429.0),        sqrt(143.0),         sqrt(13.0),                1.0,         sqrt(13.0),    sqrt(143.0/3.0),    sqrt(429.0/5.0),    sqrt(429.0/5.0),    sqrt(143.0/3.0),         sqrt(13.0),                1.0, },
    { 1.0,         sqrt(15.0),         sqrt(15.0),         sqrt(65.0),        sqrt(195.0),         sqrt(65.0),        sqrt(143.0),        sqrt(715.0),        sqrt(715.0),        sqrt(143.0),   sqrt(1287.0/7.0),       sqrt(1287.0),       sqrt(2145.0),       sqrt(1287.0),   sqrt(1287.0/7.0),        sqrt(143.0),       sqrt(1287.0),       sqrt(3003.0),       sqrt(3003.0),       sqrt(1287.0),        sqrt(143.0),         sqrt(65.0),        sqrt(715.0),       sqrt(2145.0),       sqrt(3003.0),       sqrt(2145.0),        sqrt(715.0),         sqrt(65.0),         sqrt(15.0),        sqrt(195.0),        sqrt(715.0),       sqrt(1287.0),       sqrt(1287.0),        sqrt(715.0),        sqrt(195.0),         sqrt(15.0),                1.0,         sqrt(15.0),         sqrt(65.0),        sqrt(143.0),   sqrt(1287.0/7.0),        sqrt(143.0),         sqrt(65.0),         sqrt(15.0),                1.0, },
    { 1.0,         sqrt(17.0),         sqrt(17.0),         sqrt(85.0),        sqrt(255.0),         sqrt(85.0),        sqrt(221.0),       sqrt(1105.0),       sqrt(1105.0),        sqrt(221.0),   sqrt(2431.0/7.0),       sqrt(2431.0),  sqrt(12155.0/3.0),       sqrt(2431.0),   sqrt(2431.0/7.0),   sqrt(2431.0/7.0),  sqrt(21879.0/7.0),       sqrt(7293.0),       sqrt(7293.0),  sqrt(21879.0/7.0),   sqrt(2431.0/7.0),        sqrt(221.0),       sqrt(2431.0),       sqrt(7293.0),  sqrt(51051.0/5.0),       sqrt(7293.0),       sqrt(2431.0),        sqrt(221.0),         sqrt(85.0),       sqrt(1105.0),  sqrt(12155.0/3.0),       sqrt(7293.0),       sqrt(7293.0),  sqrt(12155.0/3.0),       sqrt(1105.0),         sqrt(85.0),         sqrt(17.0),        sqrt(255.0),       sqrt(1105.0),       sqrt(2431.0),  sqrt(21879.0/7.0),       sqrt(2431.0),       sqrt(1105.0),        sqrt(255.0),         sqrt(17.0),                1.0,         sqrt(17.0),         sqrt(85.0),        sqrt(221.0),   sqrt(2431.0/7.0),   sqrt(2431.0/7.0),        sqrt(221.0),         sqrt(85.0),         sqrt(17.0),                1.0, },
};

/*
    We've nomalized each shell according to the normalization of pure $x$ component (px, dx^2, fx^3, etc.)
    For other components, the additional normalization of the following form will apply:
    $$\sqrt{ \frac{i_x! i_y! i_z!}{(2i_x)! (2i_y)! (2i_z)!} \frac{(2L)!}{L!} }$$
*/
template<int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_AUX_L && j_L >= 0 && j_L <= MAX_AUX_L)
static void kernel_cartesian_normalize_2d(double cartesian_matrix[CARTESIAN_ORBITAL_COUNT(i_L) * CARTESIAN_ORBITAL_COUNT(j_L)])
{
    if constexpr (i_L <= 1 && j_L <= 1)
        return;

    constexpr int n_density_i = CARTESIAN_ORBITAL_COUNT(i_L);
    constexpr int n_density_j = CARTESIAN_ORBITAL_COUNT(j_L);
#pragma GCC ivdep
    for (int i = 0; i < n_density_i; i++)
    {
#pragma GCC ivdep
        for (int j = 0; j < n_density_j; j++)
        {
            cartesian_matrix[i * n_density_j + j] *= kernel_cartesian_normalization_constants[i_L][i] * kernel_cartesian_normalization_constants[j_L][j];
        }
    }
}

template<int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_AUX_L && j_L >= 0 && j_L <= MAX_AUX_L)
static void kernel_cartesian_normalize_2d(double cartesian_matrix[CARTESIAN_ORBITAL_COUNT(i_L) * CARTESIAN_ORBITAL_COUNT(j_L)],
                                          const int i_begin, const int i_end)
{
    if constexpr (i_L <= 1 && j_L <= 1)
        return;

    constexpr int n_density_j = CARTESIAN_ORBITAL_COUNT(j_L);
#pragma GCC ivdep
    for (int i = i_begin; i < i_end; i++)
    {
#pragma GCC ivdep
        for (int j = 0; j < n_density_j; j++)
        {
            cartesian_matrix[(i - i_begin) * n_density_j + j] *= kernel_cartesian_normalization_constants[i_L][i] * kernel_cartesian_normalization_constants[j_L][j];
        }
    }
}

template<int i_L, int increment> requires (i_L >= 0 && i_L <= MAX_AUX_L)
static void kernel_cartesian_normalize_1d(double* cartesian_matrix,
                                          const int i_begin, const int i_end)
{
    if constexpr (i_L <= 1)
        return;

#pragma GCC ivdep
    for (int i = i_begin; i < i_end; i++)
    {
        cartesian_matrix[(i - i_begin) * increment] *= kernel_cartesian_normalization_constants[i_L][i];
    }
}
