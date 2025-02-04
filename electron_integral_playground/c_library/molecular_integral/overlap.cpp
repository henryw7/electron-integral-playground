
#include <math.h>
#include <stdio.h>

#include "../mcmurchie_davidson_E_term.hpp"
#include "../kernel_cartesian_normalization.hpp"
#include "../cartesian_spherical_transformation.hpp"
#include "../parallel_utility.h"

/*
    \begin{align*}
    S_{\mu\nu} &= \iiint_\infty d\vec{r} \ \mu(\vec{r}) \nu(\vec{r}) \\
        &= C_\mu C_\nu E^{i_x, j_x}_{0, x} E^{i_y, j_y}_{0, y} E^{i_z, j_z}_{0, z} \left( \frac{\pi}{p} \right)^{3/2}
    \end{align*}
*/
template <int i_L, int j_L>
static void overlap_general_kernel(const double A_a[4],
                                   const double B_b[4],
                                   const double coefficient,
                                   double S_cartesian[cartesian_orbital_total<i_L> * cartesian_orbital_total<j_L>])
{
    const double p = A_a[3] + B_b[3];
    const double minus_b_over_p = -B_b[3] / p;
    const double ABx = A_a[0] - B_b[0];
    const double ABy = A_a[1] - B_b[1];
    const double ABz = A_a[2] - B_b[2];
    const double PAx = minus_b_over_p * ABx;
    const double PAy = minus_b_over_p * ABy;
    const double PAz = minus_b_over_p * ABz;
    const double one_over_two_p = 0.5 / p;

    double E_x_ij_0[(i_L + 1) * (j_L + 1)] {NAN};
    {
        double E_x_i0_t[lower_triangular_upper_anti_triangular_total<i_L + j_L>] {NAN};
        mcmurchie_davidson_form_E_i0_0<i_L + j_L>(PAx, one_over_two_p, E_x_i0_t);
        mcmurchie_davidson_E_i0_0_to_E_ij_0<i_L, j_L>(ABx, E_x_i0_t, E_x_ij_0);
    }
    double E_y_ij_0[(i_L + 1) * (j_L + 1)] {NAN};
    {
        double E_y_i0_t[lower_triangular_upper_anti_triangular_total<i_L + j_L>] {NAN};
        mcmurchie_davidson_form_E_i0_0<i_L + j_L>(PAy, one_over_two_p, E_y_i0_t);
        mcmurchie_davidson_E_i0_0_to_E_ij_0<i_L, j_L>(ABy, E_y_i0_t, E_y_ij_0);
    }
    double E_z_ij_0[(i_L + 1) * (j_L + 1)] {NAN};
    {
        double E_z_i0_t[lower_triangular_upper_anti_triangular_total<i_L + j_L>] {NAN};
        mcmurchie_davidson_form_E_i0_0<i_L + j_L>(PAz, one_over_two_p, E_z_i0_t);
        mcmurchie_davidson_E_i0_0_to_E_ij_0<i_L, j_L>(ABz, E_z_i0_t, E_z_ij_0);
    }

#pragma unroll
    for (int i_x = i_L; i_x >= 0; i_x--)
    {
#pragma unroll
        for (int i_y = i_L - i_x; i_y >= 0; i_y--)
        {
            const int i_z = i_L - i_x - i_y;
            const int i_density = cartesian_orbital_index<i_L>(i_x, i_y);

#pragma unroll
            for (int j_x = j_L; j_x >= 0; j_x--)
            {
#pragma unroll
                for (int j_y = j_L - j_x; j_y >= 0; j_y--)
                {
                    const int j_z = j_L - j_x - j_y;
                    const int j_density = cartesian_orbital_index<j_L>(j_x, j_y);

                    const double E_ij_xyz = E_x_ij_0[i_x * (j_L + 1) + j_x] * E_y_ij_0[i_y * (j_L + 1) + j_y] * E_z_ij_0[i_z * (j_L + 1) + j_z];
                    constexpr int n_density_j = cartesian_orbital_total<j_L>;
                    S_cartesian[i_density * n_density_j + j_density] = coefficient * E_ij_xyz * pow(M_PI / p, 1.5);
                }
            }
        }
    }

    kernel_cartesian_normalize<i_L, j_L>(S_cartesian);
}

template <int i_L, int j_L>
static void overlap_general_kernel_wrapper(const int i_pair,
                                           const double* pair_A_a,
                                           const double* pair_B_b,
                                           const double* pair_coefficient,
                                           const int* pair_i_ao_start,
                                           const int* pair_j_ao_start,
                                           double* S_matrix,
                                           const int n_ao,
                                           const bool spherical)
{
    constexpr int n_density_i = cartesian_orbital_total<i_L>;
    constexpr int n_density_j = cartesian_orbital_total<j_L>;
    double S_cartesian[n_density_i * n_density_j] {NAN};
    const double A_a[4] { pair_A_a[i_pair * 4 + 0], pair_A_a[i_pair * 4 + 1], pair_A_a[i_pair * 4 + 2], pair_A_a[i_pair * 4 + 3], };
    const double B_b[4] { pair_B_b[i_pair * 4 + 0], pair_B_b[i_pair * 4 + 1], pair_B_b[i_pair * 4 + 2], pair_B_b[i_pair * 4 + 3], };
    const double coefficient = pair_coefficient[i_pair];

    overlap_general_kernel<i_L, j_L>(A_a, B_b, coefficient, S_cartesian);

    const int i_ao_start = pair_i_ao_start[i_pair];
    const int j_ao_start = pair_j_ao_start[i_pair];
    if (spherical)
    {
        cartesian_to_spherical_inplace<i_L, j_L>(S_cartesian);
        for (int i = 0; i < 2 * i_L + 1; i++)
        {
            for (int j = 0; j < 2 * j_L + 1; j++)
            {
                atomic_add(S_matrix + ((i_ao_start + i) * n_ao + (j_ao_start + j)), S_cartesian[i * n_density_j + j]);
                if (i_ao_start != j_ao_start)
                    atomic_add(S_matrix + ((i_ao_start + i) + (j_ao_start + j) * n_ao), S_cartesian[i * n_density_j + j]);
            }
        }
    }
    else
    {
        for (int i = 0; i < n_density_i; i++)
        {
            for (int j = 0; j < n_density_j; j++)
            {
                atomic_add(S_matrix + ((i_ao_start + i) * n_ao + (j_ao_start + j)), S_cartesian[i * n_density_j + j]);
                if (i_ao_start != j_ao_start)
                    atomic_add(S_matrix + ((i_ao_start + i) + (j_ao_start + j) * n_ao), S_cartesian[i * n_density_j + j]);
            }
        }
    }
}

template <int i_L, int j_L>
static void overlap_general_caller(const double* pair_A_a,
                                   const double* pair_B_b,
                                   const double* pair_coefficient,
                                   const int* pair_i_ao_start,
                                   const int* pair_j_ao_start,
                                   const int n_pair,
                                   double* S_matrix,
                                   const int n_ao,
                                   const bool spherical)
{
    for (int i_pair = 0; i_pair < n_pair; i_pair++)
    {
        overlap_general_kernel_wrapper<i_L, j_L>(i_pair, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, S_matrix, n_ao, spherical);
    }
}

extern "C"
{
    int overlap(const int i_L,
                const int j_L,
                const double* pair_A_a,
                const double* pair_B_b,
                const double* pair_coefficient,
                const int* pair_i_ao_start,
                const int* pair_j_ao_start,
                const int n_pair,
                double* S_matrix,
                const int n_ao,
                const bool spherical)
    {
        const int ij_L = i_L * 100 + j_L;
        switch (ij_L)
        {
            case   0: overlap_general_caller<0, 0>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case   1: overlap_general_caller<0, 1>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 101: overlap_general_caller<1, 1>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case   2: overlap_general_caller<0, 2>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 102: overlap_general_caller<1, 2>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 202: overlap_general_caller<2, 2>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case   3: overlap_general_caller<0, 3>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 103: overlap_general_caller<1, 3>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 203: overlap_general_caller<2, 3>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 303: overlap_general_caller<3, 3>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case   4: overlap_general_caller<0, 4>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 104: overlap_general_caller<1, 4>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 204: overlap_general_caller<2, 4>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 304: overlap_general_caller<3, 4>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 404: overlap_general_caller<4, 4>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case   5: overlap_general_caller<0, 5>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 105: overlap_general_caller<1, 5>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 205: overlap_general_caller<2, 5>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 305: overlap_general_caller<3, 5>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 405: overlap_general_caller<4, 5>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 505: overlap_general_caller<5, 5>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case   6: overlap_general_caller<0, 6>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 106: overlap_general_caller<1, 6>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 206: overlap_general_caller<2, 6>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 306: overlap_general_caller<3, 6>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 406: overlap_general_caller<4, 6>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 506: overlap_general_caller<5, 6>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            case 606: overlap_general_caller<6, 6>(pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, S_matrix, n_ao, spherical); break;
            default:
                printf("%s function does not support angular i_L = %d, j_L = %d\n", __func__ , i_L, j_L);
                fflush(stdout);
                return 1;
        }
        return 0;
    }
}
