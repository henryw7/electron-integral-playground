
#include <math.h>
#include <stdio.h>

#include "../mcmurchie_davidson_E_term.hpp"
#include "../mcmurchie_davidson_R_term.hpp"
#include "../kernel_cartesian_normalization.hpp"
#include "../cartesian_spherical_transformation.hpp"
#include "../parallel_utility.h"

/*
    \begin{align*}
    V_{\mu\nu C} &= \iiint_{\infty} d\vec{r} \ \mu(\vec{r})\nu(\vec{r}) \frac{1}{\left| \vec{r} - \vec{C} \right|} \\
        &= C_\mu C_\nu \frac{2\pi}{p} \sum^{i_x + j_x}_{t_x = 0} E^{i_x, j_x}_{t_x, x} \sum^{i_y + j_y}_{t_y = 0} E^{i_y, j_y}_{t_y, y} \sum^{i_z + j_z}_{t_z = 0} E^{i_z, j_z}_{t_z, z} R_{t_x,t_y,t_z}^0 \left( p, \vec{P} - \vec{C} \right)
    \end{align*}
*/
template <int i_L, int j_L>
static void nuclear_attraction_general_kernel(const double P_p[4],
                                              const double A_a[4],
                                              const double B_b[4],
                                              const double coefficient,
                                              const double C[4],
                                              double V_cartesian[lower_triangular_total<i_L> * lower_triangular_total<j_L>])
{
    const double p = P_p[3];
    const double ABx = A_a[0] - B_b[0];
    const double ABy = A_a[1] - B_b[1];
    const double ABz = A_a[2] - B_b[2];
    const double PAx = P_p[0] - A_a[0];
    const double PAy = P_p[1] - A_a[1];
    const double PAz = P_p[2] - A_a[2];
    const double PCx = P_p[0] - C[0];
    const double PCy = P_p[1] - C[1];
    const double PCz = P_p[2] - C[2];
    const double PC_2 = PCx * PCx + PCy * PCy + PCz * PCz;
    const double one_over_two_p = 0.5 / p;

    double E_x_ij_t[mcmurchie_davidson_E_ijt_total<i_L, j_L>] {NAN};
    {
        double E_x_i0_t[lower_triangular_total<i_L + j_L>] {NAN};
        mcmurchie_davidson_form_E_i0_t<i_L + j_L>(PAx, one_over_two_p, E_x_i0_t);
        mcmurchie_davidson_E_i0_t_to_E_ij_t<i_L, j_L>(ABx, E_x_i0_t, E_x_ij_t);
    }
    double E_y_ij_t[mcmurchie_davidson_E_ijt_total<i_L, j_L>] {NAN};
    {
        double E_y_i0_t[lower_triangular_total<i_L + j_L>] {NAN};
        mcmurchie_davidson_form_E_i0_t<i_L + j_L>(PAy, one_over_two_p, E_y_i0_t);
        mcmurchie_davidson_E_i0_t_to_E_ij_t<i_L, j_L>(ABy, E_y_i0_t, E_y_ij_t);
    }
    double E_z_ij_t[mcmurchie_davidson_E_ijt_total<i_L, j_L>] {NAN};
    {
        double E_z_i0_t[lower_triangular_total<i_L + j_L>] {NAN};
        mcmurchie_davidson_form_E_i0_t<i_L + j_L>(PAz, one_over_two_p, E_z_i0_t);
        mcmurchie_davidson_E_i0_t_to_E_ij_t<i_L, j_L>(ABz, E_z_i0_t, E_z_ij_t);
    }

    double R_xyz_0[triple_lower_triangular_total<i_L + j_L>] {NAN};
    {
        double R_000_m[i_L + j_L + 1] {NAN};
        mcmurchie_davidson_form_R_000_m<i_L + j_L>(p, PC_2, R_000_m);
        mcmurchie_davidson_R_000_m_to_R_xyz_0<i_L + j_L>(PCx, PCy, PCz, R_000_m, R_xyz_0);
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

                    double V_ij_xyz = 0.0;
                    for (int t_x = 0; t_x <= i_x + j_x; t_x++)
                    {
                        for (int t_y = 0; t_y <= i_y + j_y; t_y++)
                        {
                            for (int t_z = 0; t_z <= i_z + j_z; t_z++)
                            {
                                V_ij_xyz +=   E_x_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i_x, j_x, t_x)]
                                            * E_y_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i_y, j_y, t_y)]
                                            * E_z_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i_z, j_z, t_z)]
                                            * R_xyz_0[triple_lower_triangular_index<i_L + j_L>(t_x, t_y, t_z)];
                            }
                        }
                    }
                    V_ij_xyz *= 2.0 * M_PI / p * coefficient;
                    constexpr int n_density_j = lower_triangular_total<j_L>;
                    V_cartesian[i_density * n_density_j + j_density] = V_ij_xyz;
                }
            }
        }
    }

    kernel_cartesian_normalize<i_L, j_L>(V_cartesian);
}

template <int i_L, int j_L>
static void nuclear_attraction_general_kernel_wrapper(const int i_pair,
                                                      const int i_charge,
                                                      const double* pair_P_p,
                                                      const double* pair_A_a,
                                                      const double* pair_B_b,
                                                      const double* pair_coefficient,
                                                      const int* pair_i_ao_start,
                                                      const int* pair_j_ao_start,
                                                      const double* charge_position,
                                                      const int n_charge,
                                                      double* V_tensor,
                                                      const int n_ao,
                                                      const bool spherical)
{
    constexpr int n_density_i = lower_triangular_total<i_L>;
    constexpr int n_density_j = lower_triangular_total<j_L>;
    double V_cartesian[n_density_i * n_density_j] {NAN};
    const double P_p[4] { pair_P_p[i_pair * 4 + 0], pair_P_p[i_pair * 4 + 1], pair_P_p[i_pair * 4 + 2], pair_P_p[i_pair * 4 + 3], };
    const double A_a[4] { pair_A_a[i_pair * 4 + 0], pair_A_a[i_pair * 4 + 1], pair_A_a[i_pair * 4 + 2], pair_A_a[i_pair * 4 + 3], };
    const double B_b[4] { pair_B_b[i_pair * 4 + 0], pair_B_b[i_pair * 4 + 1], pair_B_b[i_pair * 4 + 2], pair_B_b[i_pair * 4 + 3], };
    const double coefficient = pair_coefficient[i_pair];

    const double C[3] { charge_position[i_charge * 3 + 0], charge_position[i_charge * 3 + 1], charge_position[i_charge * 3 + 2], };

    nuclear_attraction_general_kernel<i_L, j_L>(P_p, A_a, B_b, coefficient, C, V_cartesian);

    const int i_ao_start = pair_i_ao_start[i_pair];
    const int j_ao_start = pair_j_ao_start[i_pair];
    if (spherical)
    {
        cartesian_to_spherical_inplace<i_L, j_L>(V_cartesian);
        for (int i = 0; i < 2 * i_L + 1; i++)
        {
            for (int j = 0; j < 2 * j_L + 1; j++)
            {
                atomic_add(V_tensor + (((i_ao_start + i) * n_ao + (j_ao_start + j)) * n_charge + i_charge), V_cartesian[i * n_density_j + j]);
                if (i_ao_start != j_ao_start)
                    atomic_add(V_tensor + (((i_ao_start + i) + (j_ao_start + j) * n_ao) * n_charge + i_charge), V_cartesian[i * n_density_j + j]);
            }
        }
    }
    else
    {
        for (int i = 0; i < n_density_i; i++)
        {
            for (int j = 0; j < n_density_j; j++)
            {
                atomic_add(V_tensor + (((i_ao_start + i) * n_ao + (j_ao_start + j)) * n_charge + i_charge), V_cartesian[i * n_density_j + j]);
                if (i_ao_start != j_ao_start)
                    atomic_add(V_tensor + (((i_ao_start + i) + (j_ao_start + j) * n_ao) * n_charge + i_charge), V_cartesian[i * n_density_j + j]);
            }
        }
    }
}

template <int i_L, int j_L>
static void nuclear_attraction_general_caller(const double* pair_P_p,
                                              const double* pair_A_a,
                                              const double* pair_B_b,
                                              const double* pair_coefficient,
                                              const int* pair_i_ao_start,
                                              const int* pair_j_ao_start,
                                              const int n_pair,
                                              const double* charge_position,
                                              const int n_charge,
                                              double* V_tensor,
                                              const int n_ao,
                                              const bool spherical)
{
    for (int i_pair = 0; i_pair < n_pair; i_pair++)
    {
        for (int i_charge = 0; i_charge < n_charge; i_charge++)
        {
            nuclear_attraction_general_kernel_wrapper<i_L, j_L>(i_pair, i_charge, pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, charge_position, n_charge, V_tensor, n_ao, spherical);
        }
    }
}

extern "C"
{
    int nuclear_attraction(const int i_L,
                           const int j_L,
                           const double* pair_P_p,
                           const double* pair_A_a,
                           const double* pair_B_b,
                           const double* pair_coefficient,
                           const int* pair_i_ao_start,
                           const int* pair_j_ao_start,
                           const int n_pair,
                           const double* charge_position,
                           const int n_charge,
                           double* V_tensor,
                           const int n_ao,
                           const bool spherical)
    {
        const int ij_L = i_L * 100 + j_L;
        switch (ij_L)
        {
            case   0: nuclear_attraction_general_caller<0, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case   1: nuclear_attraction_general_caller<0, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 101: nuclear_attraction_general_caller<1, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case   2: nuclear_attraction_general_caller<0, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 102: nuclear_attraction_general_caller<1, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 202: nuclear_attraction_general_caller<2, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case   3: nuclear_attraction_general_caller<0, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 103: nuclear_attraction_general_caller<1, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 203: nuclear_attraction_general_caller<2, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 303: nuclear_attraction_general_caller<3, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case   4: nuclear_attraction_general_caller<0, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 104: nuclear_attraction_general_caller<1, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 204: nuclear_attraction_general_caller<2, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 304: nuclear_attraction_general_caller<3, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 404: nuclear_attraction_general_caller<4, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case   5: nuclear_attraction_general_caller<0, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 105: nuclear_attraction_general_caller<1, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 205: nuclear_attraction_general_caller<2, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 305: nuclear_attraction_general_caller<3, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 405: nuclear_attraction_general_caller<4, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 505: nuclear_attraction_general_caller<5, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case   6: nuclear_attraction_general_caller<0, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 106: nuclear_attraction_general_caller<1, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 206: nuclear_attraction_general_caller<2, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 306: nuclear_attraction_general_caller<3, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 406: nuclear_attraction_general_caller<4, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 506: nuclear_attraction_general_caller<5, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            case 606: nuclear_attraction_general_caller<6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair, charge_position, n_charge, V_tensor, n_ao, spherical); break;
            default:
                printf("%s function does not support angular i_L = %d, j_L = %d\n", __func__ , i_L, j_L);
                fflush(stdout);
                return 1;
        }
        return 0;
    }
}
