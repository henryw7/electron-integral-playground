
#pragma once

#include "math_constants.hpp"
#include "indexing.hpp"
#include "boys_function.hpp"

/*
    This function apply the vertical recurrence relation (assuming $j = 0$)
    \begin{align*}
        E_t^{i+1,0} &= \frac{1}{2p} E_{t-1}^{i,0} + (P_\tau - A_\tau)E_t^{i,0} + (t+1)E_{t+1}^{i,0} \\
        E_0^{0,0} &= 1 \\
        E_t^{i,0} &= 0 \quad if \quad t < 0 \ or \ t > i
    \end{align*}
    of the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients
    \begin{align*}
        E_t^{i+1,j} &= \frac{1}{2p} E_{t-1}^{i,j} + (P_\tau - A_\tau)E_t^{i,j} + (t+1)E_{t+1}^{i,j} \\
        E_t^{i,j+1} &= \frac{1}{2p} E_{t-1}^{i,j} + (P_\tau - B_\tau)E_t^{i,j} + (t+1)E_{t+1}^{i,j} \\
        E_0^{0,0} &= 1 \\
        E_t^{i,j} &= 0 \quad if \quad t < 0 \ or \ t > i+j
    \end{align*}
*/
template <int L>
static void mcmurchie_davidson_form_E_i0_t(const double PA, const double one_over_two_p, double E_i0_t[lower_triangular_total<L>])
{
    E_i0_t[0] = 1.0;
    if constexpr (L == 0)
        return;
    E_i0_t[1] = PA;
    E_i0_t[2] = one_over_two_p;
    if constexpr (L == 1)
        return;
    for (int i = 2; i <= L; i++)
    {
        E_i0_t[lower_triangular_index(i, 0)] = PA * E_i0_t[lower_triangular_index(i - 1, 0)] + E_i0_t[lower_triangular_index(i - 1, 1)];
#pragma unroll
        for (int t = 1; t < i - 1; t++)
        {
            E_i0_t[lower_triangular_index(i, t)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, t - 1)]
                                                   + PA * E_i0_t[lower_triangular_index(i - 1, t)]
                                                   + (t + 1) * E_i0_t[lower_triangular_index(i - 1, t + 1)];
        }
        E_i0_t[lower_triangular_index(i, i - 1)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, i - 2)] + PA * E_i0_t[lower_triangular_index(i - 1, i - 1)];
        E_i0_t[lower_triangular_index(i, i)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, i - 1)];
    }
}

/*
    This function apply the horizontal recurrence relation of the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients.
    $$E_t^{i,j+1} = E_t^{i+1,j} + (A_\tau - B_\tau) E_t^{i,j}$$
    The close form of the horizontal recurrence relation is
    $$E_t^{i,j} = \sum_{m = 0}^{j} \left({\begin{array}{*{20}c} j \\ j - m \end{array}}\right) (A_\tau - B_\tau)^{j - m} E_t^{i+m,0}$$
    Since $E_t^{i,j} = 0$ if $t > i + j$,
    $$E_t^{i,j} = \sum_{m = \max(0, t - i)}^{j} \left({\begin{array}{*{20}c} j \\ j - m \end{array}}\right) (A_\tau - B_\tau)^{j - m} E_t^{i+m,0}$$
*/
template <int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_L && j_L >= 0 && j_L <= MAX_L)
static void mcmurchie_davidson_E_i0_t_to_E_ij_t(const double AB, const double E_i0_t[lower_triangular_total<i_L + j_L>], double E_ij_t[mcmurchie_davidson_E_ijt_total<i_L, j_L>])
{
#pragma unroll
    for (int i = 0; i <= i_L; i++)
    {
#pragma unroll
        for (int t = 0; t <= i; t++)
        {
            E_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i, 0, t)] = E_i0_t[lower_triangular_index(i, 0)];
        }
    }

#pragma unroll
    for (int j = 1; j <= j_L; j++)
    {
#pragma unroll
        for (int i = 0; i <= i_L; i++)
        {
#pragma unroll
            for (int t = 0; t <= i + j; t++)
            {
                double AB_power_j_minus_t = 1.0;
                double E_ij_t_temp = 0.0;
                for (int m = j; m >= MAX(0, t - i); m--)
                {
                    E_ij_t_temp += binomial_coefficients[lower_triangular_index(j, j - m)] * AB_power_j_minus_t * E_i0_t[lower_triangular_index(i + m, 0)];
                    AB_power_j_minus_t *= AB;
                }
                E_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i, j, t)] = E_ij_t_temp;
            }
        }
    }
}

/*
    This function apply the horizontal recurrence relation of the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients
    only for the special case of $t = 0$, which is required for overlap types of integrals.
*/
template <int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_L && j_L >= 0 && j_L <= MAX_L)
static void mcmurchie_davidson_E_i0_t_to_E_ij_0(const double AB, const double E_i0_t[lower_triangular_total<i_L + j_L>], double E_ij_0[(i_L + 1) * (j_L + 1)])
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
            for (int m = j; m >= 0; m--)
            {
                E_ij_0_temp += binomial_coefficients[lower_triangular_index(j, j - m)] * AB_power_j_minus_t * E_i0_t[lower_triangular_index(i + m, 0)];
                AB_power_j_minus_t *= AB;
            }
            E_ij_0[i * (j_L + 1) + j] = E_ij_0_temp;
        }
    }
}

template <int L> requires (L >= 0)
static void mcmurchie_davidson_form_R_000_m(const double pq, const double PQ_2, double R_000_m[L + 1])
{
    boys_function_evaluate<L>(pq * PQ_2, R_000_m);
    if constexpr (L == 0)
        return;
    const double minus_2_pq = -2.0 * pq;
    double minus_2_pq_power_m = minus_2_pq;
    for (int m = 1; m <= L; m++)
    {
        R_000_m[m] *= minus_2_pq_power_m;
        minus_2_pq_power_m *= minus_2_pq;
    }
}

template <int L> requires (L >= 0)
static void mcmurchie_davidson_R_000_m_to_R_xyz_0(const double PQx, const double PQy, const double PQz, const double R_000_m[L + 1], double R_xyz_0[triple_lower_triangular_total<L>])
{
#pragma unroll
    for (int t_x = 0; t_x <= L; t_x++)
    {
#pragma unroll
        for (int t_y = 0; t_x + t_y <= L; t_y++)
        {
#pragma unroll
            for (int t_z = 0; t_x + t_y + t_z <= L; t_z++)
            {
                double R_tx_ty_tz_0 = 0.0;

                const double PQx_2 = PQx * PQx;
                double PQx_power_2m_minus_t = (t_x % 2 == 0) ? 1.0 : PQx;
                for (int m_x = t_x / 2; m_x <= t_x; m_x++)
                {
                    const double constant_prefactor_x = reverse_hermite_polynomial_coefficients[lower_triangular_index(t_x, m_x)];

                    const double PQy_2 = PQy * PQy;
                    double PQy_power_2m_minus_t = (t_y % 2 == 0) ? 1.0 : PQy;
                    for (int m_y = t_y / 2; m_y <= t_y; m_y++)
                    {
                        const double constant_prefactor_y = reverse_hermite_polynomial_coefficients[lower_triangular_index(t_y, m_y)];

                        const double PQz_2 = PQz * PQz;
                        double PQz_power_2m_minus_t = (t_z % 2 == 0) ? 1.0 : PQz;
                        for (int m_z = t_z / 2; m_z <= t_z; m_z++)
                        {
                            const double constant_prefactor_z = reverse_hermite_polynomial_coefficients[lower_triangular_index(t_z, m_z)];

                            R_tx_ty_tz_0 += constant_prefactor_x * constant_prefactor_y * constant_prefactor_z
                                            * PQx_power_2m_minus_t * PQy_power_2m_minus_t * PQz_power_2m_minus_t
                                            * R_000_m[m_x + m_y + m_z];

                            PQz_power_2m_minus_t *= PQz_2;
                        }
                        PQy_power_2m_minus_t *= PQy_2;
                    }
                    PQx_power_2m_minus_t *= PQx_2;
                }

                R_xyz_0[triple_lower_triangular_index<L>(t_x, t_y, t_z)] = R_tx_ty_tz_0;
            }
        }
    }
}
