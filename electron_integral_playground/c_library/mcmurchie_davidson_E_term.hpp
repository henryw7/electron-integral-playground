
#pragma once

#include "math_constants.hpp"
#include "indexing.hpp"

/*
    This function applies the vertical recurrence relation (assuming $j = 0$)
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
#pragma GCC ivdep
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
    This function applies the vertical recurrence relation of the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients
    necessary for $E_0^{i,0}$, including $E_t^{i,0}$ with $0 \leq t \leq min(i, L - i)$, which is required for overlap types of integrals.
*/
template <int L>
static void mcmurchie_davidson_form_E_i0_0(const double PA, const double one_over_two_p, double E_i0_t[lower_triangular_upper_anti_triangular_total<L>])
{
    E_i0_t[0] = 1.0;
    if constexpr (L == 0)
        return;
    E_i0_t[1] = PA;
    if constexpr (L == 1)
        return;
    E_i0_t[2] = one_over_two_p;
    // Attention: flooring function is taken by integer division
    for (int i = 2; i <= L / 2; i++)
    {
        E_i0_t[lower_triangular_index(i, 0)] = PA * E_i0_t[lower_triangular_index(i - 1, 0)] + E_i0_t[lower_triangular_index(i - 1, 1)];
#pragma GCC ivdep
        for (int t = 1; t < i - 1; t++)
        {
            E_i0_t[lower_triangular_index(i, t)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, t - 1)]
                                                   + PA * E_i0_t[lower_triangular_index(i - 1, t)]
                                                   + (t + 1) * E_i0_t[lower_triangular_index(i - 1, t + 1)];
        }
        E_i0_t[lower_triangular_index(i, i - 1)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, i - 2)] + PA * E_i0_t[lower_triangular_index(i - 1, i - 1)];
        E_i0_t[lower_triangular_index(i, i)] = one_over_two_p * E_i0_t[lower_triangular_index(i - 1, i - 1)];
    }
    // Attention: flooring function is taken by integer division
    for (int i = L / 2 + 1; i <= L; i++)
    {
        E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i, 0)] = PA * E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i - 1, 0)]
                                                                        + E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i - 1, 1)];
        const int remove_last_element = (L % 2 == 1 && i == L / 2 + 1) ? 1 : 0;
#pragma GCC ivdep
        for (int t = 1; t <= L - i - remove_last_element; t++)
        {
            E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i, t)] = one_over_two_p * E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i - 1, t - 1)]
                                                                            + PA * E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i - 1, t)]
                                                                            + (t + 1) * E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i - 1, t + 1)];
        }
        if (remove_last_element)
        {
            E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i, L - i)] = one_over_two_p * E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i - 1, L - i - 1)]
                                                                                + PA * E_i0_t[lower_triangular_upper_anti_triangular_index<L>(i - 1, L - i)];
        }
    }
}

/*
    This function applies the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients
    for one primitive basis function instead of a pair of primitive basis functions, which is required for integrals involving auxiliary basis functions.

    In this case, $b = 0$, $\vec{P} = \vec{A}$, and the recurrence relationship simplifies to
    \begin{align*}
        E_t^{i+1, 0} &= \frac{1}{2a} E_{t-1}^{i, 0} + (t+1) E_{t+1}^{i, 0} \\
        E_0^{0,0} &= 1 \\
        E_t^{i,0} &= 0 \quad if \quad t < 0 \ or \ t > i
    \end{align*}
    Notice that if $i + t$ is odd, then for each term generated by the recursion (both $i$ and $t$ decrease by 1, or $i$ decreases by 1 and $t$ increases by 1), $i + t$ is also odd.
    This implies that, at the end of recursion where $i = 0$, $t$ is either 1 or -1, both will give $E_{t}^{0,0} = 0$. So, $E_t^{i,0} = 0$ if $i + t$ is odd.
*/
template <int L>
static void mcmurchie_davidson_form_E_i0_t_0AB(const double one_over_two_p, double E_i0_t[lower_triangular_even_total<L>])
{
    E_i0_t[0] = 1.0;
    if constexpr (L == 0)
        return;
    E_i0_t[1] = one_over_two_p;
    if constexpr (L == 1)
        return;
    for (int i = 2; i <= L; i++)
    {
        if (i % 2 == 0)
            E_i0_t[lower_triangular_even_index(i, 0)] = E_i0_t[lower_triangular_even_index(i - 1, 1)];
#pragma GCC ivdep
        for (int t = (i % 2 == 0) ? 2 : 1; t < i; t += 2)
        {
            E_i0_t[lower_triangular_even_index(i, t)] = one_over_two_p * E_i0_t[lower_triangular_even_index(i - 1, t - 1)]
                                                        + (t + 1) * E_i0_t[lower_triangular_even_index(i - 1, t + 1)];
        }
        E_i0_t[lower_triangular_even_index(i, i)] = one_over_two_p * E_i0_t[lower_triangular_even_index(i - 1, i - 1)];
    }
}

/*
    This function computes the Cartesian to Hermite Gaussian transformation coefficients
    for one primitive basis function instead of a pair of primitive basis functions, which is required for integrals involving auxiliary basis functions.

    In this case, $b = 0$, $\vec{P} = \vec{A}$, and the close form solution is
    $$E_t^{i,0} = \frac{i!}{\left(\frac{i-t}{2}\right)! t! 2^{(i-t)/2}} \left(\frac{1}{2p}\right)^{i - \frac{i-t}{2}}$$
    if $i + t$ is even, otherwise $E_t^{i, 0} = $.
*/
static double mcmurchie_davidson_compute_E_i0_t_0AB(const double one_over_two_p, const int i, const int t)
{
    if (i < 0 || t < 0 || t > i)
        return NAN;
    if ((i + t) % 2 == 1)
        return NAN;
    if (i == 0)
        return 1;
    if (i == 1)
        return one_over_two_p;
    const int half_i_minus_t = (i - t) / 2;
    const double prefactor = reverse_hermite_polynomial_coefficients[lower_triangular_index(i, i - half_i_minus_t)];
    double one_over_two_p_power_i = 1;
    for (int i_power = 0; i_power < i - half_i_minus_t; i_power++)
        one_over_two_p_power_i *= one_over_two_p;
    return prefactor * one_over_two_p_power_i;
}

/*
    This function applies the horizontal recurrence relation of the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients.
    $$E_t^{i,j+1} = E_t^{i+1,j} + (A_\tau - B_\tau) E_t^{i,j}$$
    The close form of the horizontal recurrence relation is
    $$E_t^{i,j} = \sum_{m = 0}^{j} \left({\begin{array}{*{20}c} j \\ j - m \end{array}}\right) (A_\tau - B_\tau)^{j - m} E_t^{i+m,0}$$
    Since $E_t^{i,j} = 0$ if $t > i + j$,
    $$E_t^{i,j} = \sum_{m = \max(0, t - i)}^{j} \left({\begin{array}{*{20}c} j \\ j - m \end{array}}\right) (A_\tau - B_\tau)^{j - m} E_t^{i+m,0}$$
*/
template <int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_L && j_L >= 0 && j_L <= MAX_L)
static void mcmurchie_davidson_E_i0_t_to_E_ij_t(const double AB, const double E_i0_t[lower_triangular_total<i_L + j_L>], double E_ij_t[mcmurchie_davidson_E_ijt_total<i_L, j_L>])
{
#pragma GCC ivdep
    for (int i = 0; i <= i_L; i++)
    {
#pragma GCC ivdep
        for (int t = 0; t <= i; t++)
        {
            E_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i, 0, t)] = E_i0_t[lower_triangular_index(i, t)];
        }
    }

#pragma GCC ivdep
    for (int j = 1; j <= j_L; j++)
    {
#pragma GCC ivdep
        for (int i = 0; i <= i_L; i++)
        {
#pragma GCC ivdep
            for (int t = 0; t <= i + j; t++)
            {
                double AB_power_j_minus_m = 1.0;
                double E_ij_t_temp = 0.0;
                for (int m = j; m >= MAX(0, t - i); m--)
                {
                    E_ij_t_temp += binomial_coefficients[lower_triangular_index(j, j - m)] * AB_power_j_minus_m * E_i0_t[lower_triangular_index(i + m, t)];
                    AB_power_j_minus_m *= AB;
                }
                E_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i, j, t)] = E_ij_t_temp;
            }
        }
    }
}

/*
    This function applies the horizontal recurrence relation of the Mcmurchie-Davidson recurrence relation for Cartesian to Hermite Gaussian transformation coefficients
    only for the special case of $t = 0$, which is required for overlap types of integrals.
*/
template <int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_L && j_L >= 0 && j_L <= MAX_L)
static void mcmurchie_davidson_E_i0_0_to_E_ij_0(const double AB, const double E_i0_t[lower_triangular_upper_anti_triangular_total<i_L + j_L>], double E_ij_0[(i_L + 1) * (j_L + 1)])
{
#pragma GCC ivdep
    for (int i = 0; i <= i_L; i++)
    {
        E_ij_0[i * (j_L + 1) + 0] = E_i0_t[lower_triangular_upper_anti_triangular_index<i_L + j_L>(i, 0)];
    }

#pragma GCC ivdep
    for (int j = 1; j <= j_L; j++)
    {
#pragma GCC ivdep
        for (int i = 0; i <= i_L; i++)
        {
            double AB_power_j_minus_m = 1.0;
            double E_ij_0_temp = 0.0;
            for (int m = j; m >= 0; m--)
            {
                E_ij_0_temp += binomial_coefficients[lower_triangular_index(j, j - m)] * AB_power_j_minus_m * E_i0_t[lower_triangular_upper_anti_triangular_index<i_L + j_L>(i + m, 0)];
                AB_power_j_minus_m *= AB;
            }
            E_ij_0[i * (j_L + 1) + j] = E_ij_0_temp;
        }
    }
}
