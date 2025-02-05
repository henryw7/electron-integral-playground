
#include <math.h>
#include <stdio.h>

#include "../mcmurchie_davidson_E_term.hpp"
#include "../mcmurchie_davidson_R_term.hpp"
#include "../kernel_cartesian_normalization.hpp"
#include "../cartesian_spherical_transformation.hpp"
#include "../parallel_utility.h"

/*
    \begin{align*}
    (P|Q) &= \iiint d\vec{r}_1 \iiint d\vec{r}_2 \ P(\vec{r}_1) \frac{1}{|\vec{r}_1 - \vec{r}_2|} Q(\vec{r}_2) \\
        &= \sum_{t_x=0}^{i_x} E_{t_x}^{i_x,0}(p) \sum_{t_y=0}^{i_y} E_{t_y}^{i_y,0}(p) \sum_{t_z=0}^{i_z} E_{t_z}^{i_z,0}(p) \sum_{s_x=0}^{j_x} E_{s_x}^{j_x,0}(q) \sum_{s_y=0}^{j_y} E_{s_y}^{j_y,0}(q) \sum_{s_z=0}^{j_z} E_{s_z}^{j_z,0}(q) \\
            &\quad (-1)^{s_x+s_y+s_z} \frac{2\pi^{5/2}}{pq\sqrt{p+q}} R_{t_x+s_x,t_y+s_y,t_z+s_z}^0 \left( \frac{pq}{p+q}, \vec{A}-\vec{B} \right)
    \end{align*}

    If only the long-range part is computed (omega > 0),
    \begin{align*}
    (P|Q) &= \iiint d\vec{r}_1 \iiint d\vec{r}_2 \ P(\vec{r}_1) \frac{erf(\omega |\vec{r}_1 - \vec{r}_2|)}{|\vec{r}_1 - \vec{r}_2|} Q(\vec{r}_2) \\
        &= \sum_{t_x=0}^{i_x} E_{t_x}^{i_x,0}(p) \sum_{t_y=0}^{i_y} E_{t_y}^{i_y,0}(p) \sum_{t_z=0}^{i_z} E_{t_z}^{i_z,0}(p) \sum_{s_x=0}^{j_x} E_{s_x}^{j_x,0}(q) \sum_{s_y=0}^{j_y} E_{s_y}^{j_y,0}(q) \sum_{s_z=0}^{j_z} E_{s_z}^{j_z,0}(q) \\
            &\quad (-1)^{s_x+s_y+s_z} \frac{2\pi^{5/2} \omega}{pq\sqrt{pq + (p+q)\omega^2}} R_{t_x+s_x,t_y+s_y,t_z+s_z}^0 \left( \frac{1}{\frac{1}{p} + \frac{1}{q} + \frac{1}{\omega^2}}, \vec{P}-\vec{Q} \right)
    \end{align*}
*/
template <int i_L, int j_L>
static void two_center_general_kernel(const double A_a[4],
                                      const double B_b[4],
                                      const double coefficient,
                                      double J2c_cartesian[cartesian_orbital_total<i_L> * cartesian_orbital_total<j_L>],
                                      const double omega)
{
    const double PQx = A_a[0] - B_b[0];
    const double PQy = A_a[1] - B_b[1];
    const double PQz = A_a[2] - B_b[2];
    const double PQ_2 = PQx * PQx + PQy * PQy + PQz * PQz;
    const double p = A_a[3];
    const double q = B_b[3];
    const double one_over_two_p = 0.5 / p;
    const double one_over_two_q = 0.5 / q;
    const double prefactor = (omega == 0.0) ? (coefficient * 2.0 * pow(M_PI, 2.5) / (p * q * sqrt(p + q)))
                                            : (coefficient * 2.0 * pow(M_PI, 2.5) * omega / (p * q * sqrt(p * q + (p + q) * omega * omega)));

    // double E_i0_t[lower_triangular_even_total<i_L>] {NAN};
    // mcmurchie_davidson_form_E_i0_t_0AB<i_L>(one_over_two_p, E_i0_t);
    // double E_k0_s[lower_triangular_even_total<j_L>] {NAN};
    // mcmurchie_davidson_form_E_i0_t_0AB<j_L>(one_over_two_q, E_k0_s);

    double R_xyz_0[triple_lower_triangular_total<i_L + j_L>] {NAN};
    {
        double R_000_m[i_L + j_L + 1] {NAN};
        const double pq_over_p_plus_q = (omega == 0.0) ? (p * q / (p + q))
                                                       : (1.0 / (1.0 / p + 1.0 / q + 1.0 / (omega * omega)));
        mcmurchie_davidson_form_R_000_m<i_L + j_L>(pq_over_p_plus_q, PQ_2, R_000_m);
        mcmurchie_davidson_R_000_m_to_R_xyz_0<i_L + j_L>(PQx, PQy, PQz, R_000_m, R_xyz_0);
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

                    double J2c_ij_xyz = 0.0;
                    for (int t_x = i_x % 2; t_x <= i_x; t_x += 2)
                    {
                        for (int t_y = i_y % 2; t_y <= i_y; t_y += 2)
                        {
                            for (int t_z = i_z % 2; t_z <= i_z; t_z += 2)
                            {
                                for (int s_x = j_x % 2; s_x <= j_x; s_x += 2)
                                {
                                    for (int s_y = j_y % 2; s_y <= j_y; s_y += 2)
                                    {
                                        for (int s_z = j_z % 2; s_z <= j_z; s_z += 2)
                                        {
                                            J2c_ij_xyz += ((s_x + s_y + s_z) % 2 == 0 ? 1 : -1)
                                                          * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_p, i_x, t_x)
                                                          * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_p, i_y, t_y)
                                                          * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_p, i_z, t_z)
                                                          * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_q, j_x, s_x)
                                                          * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_q, j_y, s_y)
                                                          * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_q, j_z, s_z)
                                                          * R_xyz_0[triple_lower_triangular_index<i_L + j_L>(t_x + s_x, t_y + s_y, t_z + s_z)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    constexpr int n_density_j = cartesian_orbital_total<j_L>;
                    J2c_cartesian[i_density * n_density_j + j_density] = prefactor * J2c_ij_xyz;
                }
            }
        }
    }

    kernel_cartesian_normalize<i_L, j_L>(J2c_cartesian);
}

template <int i_L, int j_L>
static void two_center_general_kernel_wrapper(const int i_aux,
                                              const int j_aux,
                                              const double* aux_A_a,
                                              const double* aux_i_coefficient,
                                              const int* aux_i_aux_start,
                                              const double* aux_B_b,
                                              const double* aux_j_coefficient,
                                              const int* aux_j_aux_start,
                                              double* J2c_matrix,
                                              const int n_aux,
                                              const bool spherical,
                                              const double omega)
{
    const int i_aux_start = aux_i_aux_start[i_aux];
    const int j_aux_start = aux_j_aux_start[j_aux];
    if (i_L == j_L && i_aux_start > j_aux_start)
        return;

    constexpr int n_density_i = cartesian_orbital_total<i_L>;
    constexpr int n_density_j = cartesian_orbital_total<j_L>;
    double J2c_cartesian[n_density_i * n_density_j] {NAN};
    const double A_a[4] { aux_A_a[i_aux * 4 + 0], aux_A_a[i_aux * 4 + 1], aux_A_a[i_aux * 4 + 2], aux_A_a[i_aux * 4 + 3], };
    const double B_b[4] { aux_B_b[j_aux * 4 + 0], aux_B_b[j_aux * 4 + 1], aux_B_b[j_aux * 4 + 2], aux_B_b[j_aux * 4 + 3], };
    const double coefficient = aux_i_coefficient[i_aux] * aux_j_coefficient[j_aux];

    two_center_general_kernel<i_L, j_L>(A_a, B_b, coefficient, J2c_cartesian, omega);

    if (spherical)
    {
        cartesian_to_spherical_inplace<i_L, j_L>(J2c_cartesian);
        for (int i = 0; i < 2 * i_L + 1; i++)
        {
            for (int j = 0; j < 2 * j_L + 1; j++)
            {
                atomic_add(J2c_matrix + ((i_aux_start + i) * n_aux + (j_aux_start + j)), J2c_cartesian[i * n_density_j + j]);
                if (i_aux_start != j_aux_start)
                    atomic_add(J2c_matrix + ((i_aux_start + i) + (j_aux_start + j) * n_aux), J2c_cartesian[i * n_density_j + j]);
            }
        }
    }
    else
    {
        for (int i = 0; i < n_density_i; i++)
        {
            for (int j = 0; j < n_density_j; j++)
            {
                atomic_add(J2c_matrix + ((i_aux_start + i) * n_aux + (j_aux_start + j)), J2c_cartesian[i * n_density_j + j]);
                if (i_aux_start != j_aux_start)
                    atomic_add(J2c_matrix + ((i_aux_start + i) + (j_aux_start + j) * n_aux), J2c_cartesian[i * n_density_j + j]);
            }
        }
    }
}

template <int i_L, int j_L>
static void two_center_general_caller(const double* aux_A_a,
                                      const double* aux_i_coefficient,
                                      const int* aux_i_aux_start,
                                      const int n_aux_i,
                                      const double* aux_B_b,
                                      const double* aux_j_coefficient,
                                      const int* aux_j_aux_start,
                                      const int n_aux_j,
                                      double* J2c_matrix,
                                      const int n_aux,
                                      const bool spherical,
                                      const double omega)
{
    for (int i_aux = 0; i_aux < n_aux_i; i_aux++)
    {
        for (int j_aux = 0; j_aux < n_aux_j; j_aux++)
        {
            two_center_general_kernel_wrapper<i_L, j_L>(i_aux, j_aux, aux_A_a, aux_i_coefficient, aux_i_aux_start, aux_B_b, aux_j_coefficient, aux_j_aux_start, J2c_matrix, n_aux, spherical, omega);
        }
    }
}

extern "C"
{
    int two_center(const int i_L,
                   const int j_L,
                   const double* aux_A_a,
                   const double* aux_i_coefficient,
                   const int* aux_i_aux_start,
                   const int n_aux_i,
                   const double* aux_B_b,
                   const double* aux_j_coefficient,
                   const int* aux_j_aux_start,
                   const int n_aux_j,
                   double* J2c_matrix,
                   const int n_aux,
                   const bool spherical,
                   const double omega)
    {
        const int ij_L = i_L * 100 + j_L;
        switch (ij_L)
        {
            case   0: two_center_general_caller<0, 0>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   1: two_center_general_caller<0, 1>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 101: two_center_general_caller<1, 1>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   2: two_center_general_caller<0, 2>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 102: two_center_general_caller<1, 2>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 202: two_center_general_caller<2, 2>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   3: two_center_general_caller<0, 3>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 103: two_center_general_caller<1, 3>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 203: two_center_general_caller<2, 3>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 303: two_center_general_caller<3, 3>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   4: two_center_general_caller<0, 4>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 104: two_center_general_caller<1, 4>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 204: two_center_general_caller<2, 4>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 304: two_center_general_caller<3, 4>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 404: two_center_general_caller<4, 4>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   5: two_center_general_caller<0, 5>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 105: two_center_general_caller<1, 5>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 205: two_center_general_caller<2, 5>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 305: two_center_general_caller<3, 5>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 405: two_center_general_caller<4, 5>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 505: two_center_general_caller<5, 5>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   6: two_center_general_caller<0, 6>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 106: two_center_general_caller<1, 6>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 206: two_center_general_caller<2, 6>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 306: two_center_general_caller<3, 6>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 406: two_center_general_caller<4, 6>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 506: two_center_general_caller<5, 6>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 606: two_center_general_caller<6, 6>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   7: two_center_general_caller<0, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 107: two_center_general_caller<1, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 207: two_center_general_caller<2, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 307: two_center_general_caller<3, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 407: two_center_general_caller<4, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 507: two_center_general_caller<5, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 607: two_center_general_caller<6, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 707: two_center_general_caller<7, 7>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   8: two_center_general_caller<0, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 108: two_center_general_caller<1, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 208: two_center_general_caller<2, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 308: two_center_general_caller<3, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 408: two_center_general_caller<4, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 508: two_center_general_caller<5, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 608: two_center_general_caller<6, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 708: two_center_general_caller<7, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 808: two_center_general_caller<8, 8>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case   9: two_center_general_caller<0, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 109: two_center_general_caller<1, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 209: two_center_general_caller<2, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 309: two_center_general_caller<3, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 409: two_center_general_caller<4, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 509: two_center_general_caller<5, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 609: two_center_general_caller<6, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 709: two_center_general_caller<7, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 809: two_center_general_caller<8, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            case 909: two_center_general_caller<9, 9>(aux_A_a, aux_i_coefficient, aux_i_aux_start, n_aux_i, aux_B_b, aux_j_coefficient, aux_j_aux_start, n_aux_j, J2c_matrix, n_aux, spherical, omega); break;
            default:
                printf("%s function does not support angular i_L = %d, j_L = %d\n", __func__ , i_L, j_L);
                fflush(stdout);
                return 1;
        }
        return 0;
    }
}
