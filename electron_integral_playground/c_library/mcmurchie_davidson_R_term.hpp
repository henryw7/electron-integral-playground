
#pragma once

#include "math_constants.hpp"
#include "indexing.hpp"
#include "boys_function.hpp"

/*
    This function computes the base case of the Mcmurchie-Davidson recurrence relation for Hermite Gaussian integrals
    $$R_{0,0,0}^m = (-2\zeta)^m F_m\left(\zeta|\vec{PQ}|^2\right)$$

    For one electron integral, $\zeta = p$, and $\vec{PQ} = \vec{P} - \vec{C}$.
    For two electron integral, $\zeta = \frac{pq}{p+q}$, and $\vec{PQ} = \vec{P} - \vec{Q}$.
*/
template <int L> requires (L >= 0)
static void mcmurchie_davidson_form_R_000_m(const double zeta, const double PQ_2, double R_000_m[L + 1])
{
    boys_function_evaluate<L>(zeta * PQ_2, R_000_m);
    if constexpr (L == 0)
        return;
    const double minus_2_zeta = -2.0 * zeta;
    double minus_2_zeta_power_m = minus_2_zeta;
    for (int m = 1; m <= L; m++)
    {
        R_000_m[m] *= minus_2_zeta_power_m;
        minus_2_zeta_power_m *= minus_2_zeta;
    }
}

/*
    This function computes the general case of the Mcmurchie-Davidson recurrence relation for Hermite Gaussian integrals,
    with the close form
    \begin{align*}
    R_{t_x, t_y, t_z}^{m_{base}} &= \sum_{m_x = ceil(t_x / 2)}^{t_x} \frac{t_x!}{(2m_x - t_x)!(t_x - m_x)! 2^{t_x - m_x}} PQ_x^{2m_x - t_x} \\
        &\quad \sum_{m_y = ceil(t_y / 2)}^{t_y} \frac{t_y!}{(2m_y - t_y)!(t_y - m_y)! 2^{t_y - m_y}} PQ_y^{2m_y - t_y} \\
        &\quad \sum_{m_z = ceil(t_z / 2)}^{t_z} \frac{t_z!}{(2m_z - t_z)!(t_z - m_z)! 2^{t_z - m_z}} PQ_z^{2m_z - t_z} R_{0,0,0}^{m_x + m_y + m_z + m_{base}}
    \end{align*}
*/
template <int L, int m_base = 0> requires (L >= 0 && m_base >= 0)
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
                for (int m_x = (t_x + 1) / 2; m_x <= t_x; m_x++)
                {
                    const double constant_prefactor_x = reverse_hermite_polynomial_coefficients[lower_triangular_index(t_x, m_x)];

                    const double PQy_2 = PQy * PQy;
                    double PQy_power_2m_minus_t = (t_y % 2 == 0) ? 1.0 : PQy;
                    for (int m_y = (t_y + 1) / 2; m_y <= t_y; m_y++)
                    {
                        const double constant_prefactor_y = reverse_hermite_polynomial_coefficients[lower_triangular_index(t_y, m_y)];

                        const double PQz_2 = PQz * PQz;
                        double PQz_power_2m_minus_t = (t_z % 2 == 0) ? 1.0 : PQz;
                        for (int m_z = (t_z + 1) / 2; m_z <= t_z; m_z++)
                        {
                            const double constant_prefactor_z = reverse_hermite_polynomial_coefficients[lower_triangular_index(t_z, m_z)];

                            R_tx_ty_tz_0 += constant_prefactor_x * constant_prefactor_y * constant_prefactor_z
                                            * PQx_power_2m_minus_t * PQy_power_2m_minus_t * PQz_power_2m_minus_t
                                            * R_000_m[m_x + m_y + m_z + m_base];

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
