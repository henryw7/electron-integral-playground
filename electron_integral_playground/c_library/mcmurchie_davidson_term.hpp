
#pragma once

#include "math_constants.hpp"
#include "indexing.hpp"

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
static void mcmurchie_davidson_form_E_i0_t(const double PA, const double one_over_two_p, double E_i0_t[(L + 1) * (L + 2) / 2])
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
    This function apply the horizontal recurrence relation of the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients
    $$E_t^{i,j+1} = E_t^{i+1,j} + (A_\tau - B_\tau) E_t^{i,j}$$
    The close form of the horizontal recurrence relation is
    $$E_t^{i,j} = \sum_{m = 0}^{j} \left({\begin{array}{*{20}c} j \\ j - m \end{array}}\right) (A_\tau - B_\tau)^{j - m} E_t^{i+m,0}$$
    Since $E_t^{i,j} = 0$ if $t > i + j$,
    $$E_t^{i,j} = \sum_{m = \max(0, t - i)}^{j} \left({\begin{array}{*{20}c} j \\ j - m \end{array}}\right) (A_\tau - B_\tau)^{j - m} E_t^{i+m,0}$$
    
    This function only implement the special case of $t = 0$, which is required for overlap types of integrals.
*/
template <int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_L && j_L >= 0 && j_L <= MAX_L)
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
            for (int m = j; m >= 0; m--)
            {
                E_ij_0_temp += binomial_coefficients[lower_triangular_index(j, j - m)] * AB_power_j_minus_t * E_i0_t[lower_triangular_index(i + m, 0)];
                AB_power_j_minus_t *= AB;
            }
            E_ij_0[i * (j_L + 1) + j] = E_ij_0_temp;
        }
    }
}
