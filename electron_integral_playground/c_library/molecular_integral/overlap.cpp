
#include <math.h>
#include <stdio.h>

static const int binomial_coefficient[16 * (16 + 1) / 2] = {
    1,
    1,   1,
    1,   2,   1,
    1,   3,   3,   1,
    1,   4,   6,   4,   1,
    1,   5,  10,  10,   5,   1,
    1,   6,  15,  20,  15,   6,   1,
    1,   7,  21,  35,  35,  21,   7,   1,
    1,   8,  28,  56,  70,  56,  28,   8,   1,
    1,   9,  36,  84, 126, 126,  84,  36,   9,   1,
    1,  10,  45, 120, 210, 252, 210, 120,  45,  10,   1,
    1,  11,  55, 165, 330, 462, 462, 330, 165,  55,  11,   1,
    1,  12,  66, 220, 495, 792, 924, 792, 495, 220,  66,  12,   1,
    1,  13,  78, 286, 715,1287,1716,1716,1287, 715, 286,  78,  13,   1,
    1,  14,  91, 364,1001,2002,3003,3432,3003,2002,1001, 364,  91,  14,   1,
    1,  15, 105, 455,1365,3003,5005,6435,6435,5005,3003,1365, 455, 105,  15,   1,
};

static inline int lower_triangular_index(const int row, const int column)
{
    return row * (row + 1) / 2 + column;
}

template <int L>
static void mcmurchie_davidson_form_E_i0_t(const double PA, const double one_over_two_p, double E_i0_t[(L + 1) * (L + 2) / 2])
{
    E_i0_t[lower_triangular_index(0, 0)] = 1.0;
    for (int i = 1; i <= L; i++)
    {
        E_i0_t[lower_triangular_index(i, 0)] = PA * E_i0_t[lower_triangular_index(i - 1, 0)] + E_i0_t[lower_triangular_index(i - 1, 1)];
#pragma unroll
        for (int t = 1; t < i; t++)
        {
            E_i0_t[lower_triangular_index(i, t)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, t - 1)]
                                                   + PA * E_i0_t[lower_triangular_index(i - 1, t)]
                                                   + (t + 1) * E_i0_t[lower_triangular_index(i - 1, t + 1)];
        }
        E_i0_t[lower_triangular_index(i, i)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, i - 1)];
    }
}

template <int i_L, int j_L>
static void mcmurchie_davidson_E_i0_t_to_E_ij_0(const double AB, const double E_i0_t[(i_L + j_L + 1) * (i_L + j_L + 2) / 2], double E_ij_0[(i_L + 1) * (j_L + 1)])
{
#pragma unroll
    for (int i = 0; i <= i_L; i++)
    {
        E_ij_0[i * (j_L + 1) + 0] = E_i0_t[lower_triangular_index(i, 0)];
    }

#pragma unroll
    for (int j = 1; j <= j_L; j++)
    {
#pragma unroll
        for (int i = 0; i <= i_L; i++)
        {
            double AB_power_j_minus_t = 1.0;
            double E_ij_0_temp = 0.0;
            for (int t = j; t >= 0; t--)
            {
                E_ij_0_temp += binomial_coefficient[lower_triangular_index(j, j - t)] * AB_power_j_minus_t * E_i0_t[lower_triangular_index(i + t, 0)];
                AB_power_j_minus_t *= AB;
            }
            E_ij_0[i * (j_L + 1) + j] = E_ij_0_temp;
        }
    }
}

template <int i_L, int j_L>
static void overlap_general_kernel(const double A_a[4],
                                   const double B_b[4],
                                   const double coefficient,
                                   double S_cartesian[(i_L + 1) * (i_L + 2) / 2 * (j_L + 1) * (j_L + 2) / 2])
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
        double E_x_i0_t[(i_L + j_L + 1) * (i_L + j_L + 2) / 2] {NAN};
        mcmurchie_davidson_form_E_i0_t<i_L + j_L>(PAx, one_over_two_p, E_x_i0_t);
        mcmurchie_davidson_E_i0_t_to_E_ij_0<i_L, j_L>(ABx, E_x_i0_t, E_x_ij_0);
    }
    double E_y_ij_0[(i_L + 1) * (j_L + 1)] {NAN};
    {
        double E_y_i0_t[(i_L + j_L + 1) * (i_L + j_L + 2) / 2] {NAN};
        mcmurchie_davidson_form_E_i0_t<i_L + j_L>(PAy, one_over_two_p, E_y_i0_t);
        mcmurchie_davidson_E_i0_t_to_E_ij_0<i_L, j_L>(ABy, E_y_i0_t, E_y_ij_0);
    }
    double E_z_ij_0[(i_L + 1) * (j_L + 1)] {NAN};
    {
        double E_z_i0_t[(i_L + j_L + 1) * (i_L + j_L + 2) / 2] {NAN};
        mcmurchie_davidson_form_E_i0_t<i_L + j_L>(PAz, one_over_two_p, E_z_i0_t);
        mcmurchie_davidson_E_i0_t_to_E_ij_0<i_L, j_L>(ABz, E_z_i0_t, E_z_ij_0);
    }

#pragma unroll
    for (int i_x = i_L; i_x >= 0; i_x--)
    {
#pragma unroll
        for (int i_y = i_L - i_x; i_y >= 0; i_y--)
        {
            const int i_z = i_L - i_x - i_y;
            const int i_density = (i_L - i_x) * (i_L - i_x + 1) / 2 + i_L - i_x - i_y;

#pragma unroll
            for (int j_x = j_L; j_x >= 0; j_x--)
            {
#pragma unroll
                for (int j_y = j_L - j_x; j_y >= 0; j_y--)
                {
                    const int j_z = j_L - j_x - j_y;
                    const int j_density = (j_L - j_x) * (j_L - j_x + 1) / 2 + j_L - j_x - j_y;

                    const double E_ij_xyz = E_x_ij_0[i_x * (j_L + 1) + j_x] * E_y_ij_0[i_y * (j_L + 1) + j_y] * E_z_ij_0[i_z * (j_L + 1) + j_z];
                    constexpr int n_density_j = (j_L + 1) * (j_L + 2) / 2;
                    S_cartesian[i_density * n_density_j + j_density] = coefficient * E_ij_xyz * pow(M_PI / p, 1.5);
                }
            }
        }
    }
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
    constexpr int n_density_i = (i_L + 1) * (i_L + 2) / 2;
    constexpr int n_density_j = (j_L + 1) * (j_L + 2) / 2;
    double S_cartesian[n_density_i * n_density_j] {NAN};
    const double A_a[4] { pair_A_a[i_pair * 4 + 0], pair_A_a[i_pair * 4 + 1], pair_A_a[i_pair * 4 + 2], pair_A_a[i_pair * 4 + 3], };
    const double B_b[4] { pair_B_b[i_pair * 4 + 0], pair_B_b[i_pair * 4 + 1], pair_B_b[i_pair * 4 + 2], pair_B_b[i_pair * 4 + 3], };
    const double coefficient = pair_coefficient[i_pair];

    overlap_general_kernel<i_L, j_L>(A_a, B_b, coefficient, S_cartesian);

    const int i_ao_start = pair_i_ao_start[i_pair];
    const int j_ao_start = pair_j_ao_start[i_pair];
    for (int i = 0; i < n_density_i; i++)
    {
        for (int j = 0; j < n_density_j; j++)
        {
            S_matrix[(i_ao_start + i) * n_ao + (j_ao_start + j)] += S_cartesian[i * n_density_j + j];
            if (i_ao_start != j_ao_start)
                S_matrix[(i_ao_start + i) + (j_ao_start + j) * n_ao] += S_cartesian[i * n_density_j + j];
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
            default:
                printf("%s function does not support angular i_L = %d, j_L = %d\n", __func__ , i_L, j_L);
                fflush(stdout);
                return 1;
        }
        return 0;
    }
}
