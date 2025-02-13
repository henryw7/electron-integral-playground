
#include <math.h>
#include <stdio.h>

#include "../mcmurchie_davidson_E_term.hpp"
#include "../mcmurchie_davidson_R_term.hpp"
#include "../kernel_cartesian_normalization.hpp"
#include "../cartesian_spherical_transformation.hpp"
#include "../parallel_utility.h"

/*
    \begin{align*}
    (P|\mu\lambda) &= \iiint d\vec{r}_1 \iiint d\vec{r}_2 \ P(\vec{r}_1) \frac{1}{|\vec{r}_1 - \vec{r}_2|} \mu(\vec{r}_2)\nu(\vec{r}_2) \\
        &= \sum_{s_x=0}^{k_x} E_{s_x}^{k_x, 0}(q) \sum_{s_y=0}^{k_y} E_{s_y}^{k_y, 0}(q) \sum_{s_z=0}^{k_z} E_{s_z}^{k_z, 0}(q) \\
            &\quad \sum_{t_x=0}^{i_x+j_x} E_{t_x}^{i_xj_x}(A_x-B_x,a,b) \sum_{t_y=0}^{i_y+j_y} E_{t_y}^{i_yj_y}(A_y-B_y,a,b) \sum_{t_z=0}^{i_z+j_z} E_{t_z}^{i_zj_z}(A_z-B_z,a,b) \\
            &\quad (-1)^{s_x+s_y+s_z} \frac{2\pi^{5/2}}{pq \sqrt{p+q}} R_{t_x+s_x, t_y+s_y, t_z+s_z}^0\left( \frac{pq}{p+q}, \vec{P} - \vec{C} \right)
    \end{align*}

    If only the long-range part is computed (omega > 0),
    \begin{align*}
    (P|\mu\lambda) &= \iiint d\vec{r}_1 \iiint d\vec{r}_2 \ P(\vec{r}_1) \frac{erf(\omega |\vec{r}_1 - \vec{r}_2|)}{|\vec{r}_1 - \vec{r}_2|} \mu(\vec{r}_2)\nu(\vec{r}_2) \\
        &= \sum_{s_x=0}^{k_x} E_{s_x}^{k_x, 0}(q) \sum_{s_y=0}^{k_y} E_{s_y}^{k_y, 0}(q) \sum_{s_z=0}^{k_z} E_{s_z}^{k_z, 0}(q) \\
            &\quad \sum_{t_x=0}^{i_x+j_x} E_{t_x}^{i_xj_x}(A_x-B_x,a,b) \sum_{t_y=0}^{i_y+j_y} E_{t_y}^{i_yj_y}(A_y-B_y,a,b) \sum_{t_z=0}^{i_z+j_z} E_{t_z}^{i_zj_z}(A_z-B_z,a,b) \\
            &\quad (-1)^{s_x+s_y+s_z} \frac{2\pi^{5/2} \omega}{pq\sqrt{pq + (p+q)\omega^2}} R_{t_x+s_x, t_y+s_y, t_z+s_z}^0\left( \frac{1}{\frac{1}{p} + \frac{1}{q} + \frac{1}{\omega^2}}, \vec{P} - \vec{C} \right)
    \end{align*}
*/
template <int i_L, int j_L, int k_L, int max_n_density_k_per_split>
static void three_center_general_kernel(const double P_p[4],
                                        const double A_a[4],
                                        const double B_b[4],
                                        const double C_q[4],
                                        const double coefficient,
                                        double J3c_cartesian[max_n_density_k_per_split * cartesian_orbital_total<i_L> * cartesian_orbital_total<j_L>],
                                        const int k_density_begin,
                                        const int k_density_end,
                                        const double omega)
{
    const double PQx = P_p[0] - C_q[0];
    const double PQy = P_p[1] - C_q[1];
    const double PQz = P_p[2] - C_q[2];
    const double PQ_2 = PQx * PQx + PQy * PQy + PQz * PQz;
    const double p = P_p[3];
    const double q = C_q[3];
    const double one_over_two_p = 0.5 / p;
    const double one_over_two_q = 0.5 / q;
    const double prefactor = (omega == 0.0) ? (coefficient * 2.0 * pow(M_PI, 2.5) / (p * q * sqrt(p + q)))
                                            : (coefficient * 2.0 * pow(M_PI, 2.5) * omega / (p * q * sqrt(p * q + (p + q) * omega * omega)));
    const double pq_over_p_plus_q = (omega == 0.0) ? (p * q / (p + q))
                                                   : (1.0 / (1.0 / p + 1.0 / q + 1.0 / (omega * omega)));

    const double ABx = A_a[0] - B_b[0];
    const double ABy = A_a[1] - B_b[1];
    const double ABz = A_a[2] - B_b[2];
    const double PAx = P_p[0] - A_a[0];
    const double PAy = P_p[1] - A_a[1];
    const double PAz = P_p[2] - A_a[2];
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

    constexpr bool store_R_xyz_0 = triple_lower_triangular_total<i_L + j_L + k_L> <= MAX_REGISTER_MATRIX_DOUBLE_COUNT;
    constexpr int R_xyz_t_size = store_R_xyz_0 ? triple_lower_triangular_total<i_L + j_L + k_L> : (i_L + j_L + k_L + 1);
    double R_xyz_t[R_xyz_t_size] {NAN};
    if constexpr (store_R_xyz_0)
    {
        double R_000_m[i_L + j_L + k_L + 1] {NAN};
        mcmurchie_davidson_form_R_000_m<i_L + j_L + k_L>(pq_over_p_plus_q, PQ_2, R_000_m);
        mcmurchie_davidson_R_000_m_to_R_xyz_0<i_L + j_L + k_L>(PQx, PQy, PQz, R_000_m, R_xyz_t);
    }
    else
    {
        mcmurchie_davidson_form_R_000_m<i_L + j_L + k_L>(pq_over_p_plus_q, PQ_2, R_xyz_t);
    }

    for (int k_density = k_density_begin; k_density < k_density_end; k_density++)
    {
        const int k_x = cartesian_orbital_index_x<k_L>(k_density);
        const int k_y = cartesian_orbital_index_y<k_L>(k_density);
        const int k_z = cartesian_orbital_index_z<k_L>(k_density);

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

                        double J3c_ijk_xyz = 0.0;
                        for (int t_x = 0; t_x <= i_x + j_x; t_x++)
                        {
                            for (int t_y = 0; t_y <= i_y + j_y; t_y++)
                            {
                                for (int t_z = 0; t_z <= i_z + j_z; t_z++)
                                {
                                    for (int s_x = k_x % 2; s_x <= k_x; s_x += 2)
                                    {
                                        for (int s_y = k_y % 2; s_y <= k_y; s_y += 2)
                                        {
                                            for (int s_z = k_z % 2; s_z <= k_z; s_z += 2)
                                            {
                                                double R_xyz_0 = NAN;
                                                if constexpr (store_R_xyz_0)
                                                    R_xyz_0 = R_xyz_t[triple_lower_triangular_index<i_L + j_L + k_L>(t_x + s_x, t_y + s_y, t_z + s_z)];
                                                else
                                                    R_xyz_0 = mcmurchie_davidson_R_000_m_to_R_xyz_0<i_L + j_L + k_L>(PQx, PQy, PQz, R_xyz_t, t_x + s_x, t_y + s_y, t_z + s_z);
                                                J3c_ijk_xyz += ((s_x + s_y + s_z) % 2 == 0 ? 1 : -1)
                                                               * E_x_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i_x, j_x, t_x)]
                                                               * E_y_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i_y, j_y, t_y)]
                                                               * E_z_ij_t[mcmurchie_davidson_E_ijt_index<j_L>(i_z, j_z, t_z)]
                                                               * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_q, k_x, s_x)
                                                               * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_q, k_y, s_y)
                                                               * mcmurchie_davidson_compute_E_i0_t_0AB(one_over_two_q, k_z, s_z)
                                                               * R_xyz_0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        constexpr int n_density_i = cartesian_orbital_total<i_L>;
                        constexpr int n_density_j = cartesian_orbital_total<j_L>;
                        J3c_cartesian[(k_density - k_density_begin) * n_density_i * n_density_j + i_density * n_density_j + j_density] = prefactor * J3c_ijk_xyz;
                    }
                }
            }
        }
    }

    constexpr int n_density_ij = cartesian_orbital_total<i_L> * cartesian_orbital_total<j_L>;
#pragma unroll
    for (int k_density = k_density_begin; k_density < k_density_end; k_density++)
    {
        kernel_cartesian_normalize_2d<i_L, j_L>(J3c_cartesian + (k_density - k_density_begin) * n_density_ij);
    }
#pragma unroll
    for (int ij = 0; ij < n_density_ij; ij++)
    {
        kernel_cartesian_normalize_1d<k_L, n_density_ij>(J3c_cartesian + ij, k_density_begin, k_density_end);
    }
}

template <int i_L, int j_L, int k_L>
static void three_center_general_kernel_wrapper(const int i_pair_ij,
                                                const int i_aux_k,
                                                const double* pair_P_p,
                                                const double* pair_A_a,
                                                const double* pair_B_b,
                                                const double* pair_coefficient,
                                                const int* pair_i_ao_start,
                                                const int* pair_j_ao_start,
                                                const double* aux_A_a,
                                                const double* aux_coefficient,
                                                const int* aux_i_aux_start,
                                                double* J3c_matrix,
                                                const int n_ao,
                                                const int n_aux,
                                                const bool spherical,
                                                const double omega)
{
    const double P_p[4] { pair_P_p[i_pair_ij * 4 + 0], pair_P_p[i_pair_ij * 4 + 1], pair_P_p[i_pair_ij * 4 + 2], pair_P_p[i_pair_ij * 4 + 3], };
    const double A_a[4] { pair_A_a[i_pair_ij * 4 + 0], pair_A_a[i_pair_ij * 4 + 1], pair_A_a[i_pair_ij * 4 + 2], pair_A_a[i_pair_ij * 4 + 3], };
    const double B_b[4] { pair_B_b[i_pair_ij * 4 + 0], pair_B_b[i_pair_ij * 4 + 1], pair_B_b[i_pair_ij * 4 + 2], pair_B_b[i_pair_ij * 4 + 3], };
    const double C_q[4] { aux_A_a[i_aux_k * 4 + 0], aux_A_a[i_aux_k * 4 + 1], aux_A_a[i_aux_k * 4 + 2], aux_A_a[i_aux_k * 4 + 3], };
    const double coefficient = pair_coefficient[i_pair_ij] * aux_coefficient[i_aux_k];

    constexpr int n_density_i = cartesian_orbital_total<i_L>;
    constexpr int n_density_j = cartesian_orbital_total<j_L>;
    constexpr int n_density_k = cartesian_orbital_total<k_L>;

    constexpr int n_density_k_split = (n_density_i * n_density_j * n_density_k + MAX_REGISTER_MATRIX_DOUBLE_COUNT - 1) / MAX_REGISTER_MATRIX_DOUBLE_COUNT;
    constexpr int max_n_density_k_per_split = (n_density_k + n_density_k_split - 1) / n_density_k_split;

    const int i_ao_start = pair_i_ao_start[i_pair_ij];
    const int j_ao_start = pair_j_ao_start[i_pair_ij];
    const int k_aux_start = aux_i_aux_start[i_aux_k];
    for (int k_density_begin = 0; k_density_begin < n_density_k; k_density_begin += max_n_density_k_per_split)
    {
        const int k_density_end = MIN(k_density_begin + max_n_density_k_per_split, n_density_k);
        double J3c_cartesian[max_n_density_k_per_split * n_density_i * n_density_j] {NAN};

        three_center_general_kernel<i_L, j_L, k_L, max_n_density_k_per_split>(P_p, A_a, B_b, C_q, coefficient, J3c_cartesian, k_density_begin, k_density_end, omega);

        if (spherical)
        {
            constexpr int n_density_ij = n_density_i * n_density_j;
#pragma unroll
            for (int k = k_density_begin; k < k_density_end; k++)
            {
                cartesian_to_spherical_2d_inplace<i_L, j_L>(J3c_cartesian + (k - k_density_begin) * n_density_ij);
            }

            if constexpr (n_density_k_split == 1)
            {
#pragma unroll
                for (int i = 0; i < 2 * i_L + 1; i++)
                {
#pragma unroll
                    for (int j = 0; j < 2 * j_L + 1; j++)
                    {
                        cartesian_to_spherical_1d_inplace<k_L, n_density_ij>(J3c_cartesian + (i * n_density_j + j));
                    }
                }
                for (int k = 0; k < 2 * k_L + 1; k++)
                {
                    for (int i = 0; i < 2 * i_L + 1; i++)
                    {
                        for (int j = 0; j < 2 * j_L + 1; j++)
                        {
                            atomic_add(J3c_matrix + ((k_aux_start + k) * n_ao * n_ao + (i_ao_start + i) * n_ao + (j_ao_start + j)), J3c_cartesian[k * n_density_ij + i * n_density_j + j]);
                            if (i_ao_start != j_ao_start)
                                atomic_add(J3c_matrix + ((k_aux_start + k) * n_ao * n_ao + (i_ao_start + i) + (j_ao_start + j) * n_ao), J3c_cartesian[k * n_density_ij + i * n_density_j + j]);
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < 2 * i_L + 1; i++)
                {
                    for (int j = 0; j < 2 * j_L + 1; j++)
                    {
                        double J3c_spherical[2 * k_L + 1] {0.0};
                        for (int k = k_density_begin; k < k_density_end; k++)
                        {
                            cartesian_to_spherical_1d_scatter<k_L, 1>(J3c_cartesian[(k - k_density_begin) * n_density_ij + i * n_density_j + j], k, J3c_spherical);
                        }

                        for (int k = 0; k < 2 * k_L + 1; k++)
                        {
                            if (J3c_spherical[k] == 0.0)
                                continue;
                            atomic_add(J3c_matrix + ((k_aux_start + k) * n_ao * n_ao + (i_ao_start + i) * n_ao + (j_ao_start + j)), J3c_spherical[k]);
                            if (i_ao_start != j_ao_start)
                                atomic_add(J3c_matrix + ((k_aux_start + k) * n_ao * n_ao + (i_ao_start + i) + (j_ao_start + j) * n_ao), J3c_spherical[k]);
                        }
                    }
                }
            }
        }
        else
        {
            for (int k = k_density_begin; k < k_density_end; k++)
            {
                for (int i = 0; i < n_density_i; i++)
                {
                    for (int j = 0; j < n_density_j; j++)
                    {
                        atomic_add(J3c_matrix + ((k_aux_start + k) * n_ao * n_ao + (i_ao_start + i) * n_ao + (j_ao_start + j)), J3c_cartesian[(k - k_density_begin) * n_density_i * n_density_j + i * n_density_j + j]);
                        if (i_ao_start != j_ao_start)
                            atomic_add(J3c_matrix + ((k_aux_start + k) * n_ao * n_ao + (i_ao_start + i) + (j_ao_start + j) * n_ao), J3c_cartesian[(k - k_density_begin) * n_density_i * n_density_j + i * n_density_j + j]);
                    }
                }
            }
        }
    }
}

template <int i_L, int j_L, int k_L>
static void three_center_general_caller(const double* pair_P_p,
                                        const double* pair_A_a,
                                        const double* pair_B_b,
                                        const double* pair_coefficient,
                                        const int* pair_i_ao_start,
                                        const int* pair_j_ao_start,
                                        const int n_pair_ij,
                                        const double* aux_A_a,
                                        const double* aux_coefficient,
                                        const int* aux_i_aux_start,
                                        const int n_aux_primitive_k,
                                        double* J3c_matrix,
                                        const int n_ao,
                                        const int n_aux,
                                        const bool spherical,
                                        const double omega)
{
    for (int i_pair_ij = 0; i_pair_ij < n_pair_ij; i_pair_ij++)
    {
        for (int i_aux_k = 0; i_aux_k < n_aux_primitive_k; i_aux_k++)
        {
            three_center_general_kernel_wrapper<i_L, j_L, k_L>(i_pair_ij, i_aux_k, pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, aux_A_a, aux_coefficient, aux_i_aux_start, J3c_matrix, n_ao, n_aux, spherical, omega);
        }
    }
}

extern "C"
{
    int three_center(const int i_L,
                     const int j_L,
                     const int k_L,
                     const double* pair_P_p,
                     const double* pair_A_a,
                     const double* pair_B_b,
                     const double* pair_coefficient,
                     const int* pair_i_ao_start,
                     const int* pair_j_ao_start,
                     const int n_pair_ij,
                     const double* aux_A_a,
                     const double* aux_coefficient,
                     const int* aux_i_aux_start,
                     const int n_aux_primitive_k,
                     double* J3c_matrix,
                     const int n_ao,
                     const int n_aux,
                     const bool spherical,
                     const double omega)
    {
        const int kij_L = k_L * 10000 + i_L * 100 + j_L;
        switch (kij_L)
        {
            case     0: three_center_general_caller<0, 0, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case     1: three_center_general_caller<0, 1, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   101: three_center_general_caller<1, 1, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case     2: three_center_general_caller<0, 2, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   102: three_center_general_caller<1, 2, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   202: three_center_general_caller<2, 2, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case     3: three_center_general_caller<0, 3, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   103: three_center_general_caller<1, 3, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   203: three_center_general_caller<2, 3, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   303: three_center_general_caller<3, 3, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case     4: three_center_general_caller<0, 4, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   104: three_center_general_caller<1, 4, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   204: three_center_general_caller<2, 4, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   304: three_center_general_caller<3, 4, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   404: three_center_general_caller<4, 4, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case     5: three_center_general_caller<0, 5, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   105: three_center_general_caller<1, 5, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   205: three_center_general_caller<2, 5, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   305: three_center_general_caller<3, 5, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   405: three_center_general_caller<4, 5, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   505: three_center_general_caller<5, 5, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case     6: three_center_general_caller<0, 6, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   106: three_center_general_caller<1, 6, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   206: three_center_general_caller<2, 6, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   306: three_center_general_caller<3, 6, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   406: three_center_general_caller<4, 6, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   506: three_center_general_caller<5, 6, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case   606: three_center_general_caller<6, 6, 0>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10000: three_center_general_caller<0, 0, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10001: three_center_general_caller<0, 1, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10101: three_center_general_caller<1, 1, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10002: three_center_general_caller<0, 2, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10102: three_center_general_caller<1, 2, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10202: three_center_general_caller<2, 2, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10003: three_center_general_caller<0, 3, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10103: three_center_general_caller<1, 3, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10203: three_center_general_caller<2, 3, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10303: three_center_general_caller<3, 3, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10004: three_center_general_caller<0, 4, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10104: three_center_general_caller<1, 4, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10204: three_center_general_caller<2, 4, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10304: three_center_general_caller<3, 4, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10404: three_center_general_caller<4, 4, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10005: three_center_general_caller<0, 5, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10105: three_center_general_caller<1, 5, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10205: three_center_general_caller<2, 5, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10305: three_center_general_caller<3, 5, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10405: three_center_general_caller<4, 5, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10505: three_center_general_caller<5, 5, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10006: three_center_general_caller<0, 6, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10106: three_center_general_caller<1, 6, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10206: three_center_general_caller<2, 6, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10306: three_center_general_caller<3, 6, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10406: three_center_general_caller<4, 6, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10506: three_center_general_caller<5, 6, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 10606: three_center_general_caller<6, 6, 1>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20000: three_center_general_caller<0, 0, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20001: three_center_general_caller<0, 1, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20101: three_center_general_caller<1, 1, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20002: three_center_general_caller<0, 2, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20102: three_center_general_caller<1, 2, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20202: three_center_general_caller<2, 2, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20003: three_center_general_caller<0, 3, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20103: three_center_general_caller<1, 3, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20203: three_center_general_caller<2, 3, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20303: three_center_general_caller<3, 3, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20004: three_center_general_caller<0, 4, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20104: three_center_general_caller<1, 4, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20204: three_center_general_caller<2, 4, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20304: three_center_general_caller<3, 4, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20404: three_center_general_caller<4, 4, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20005: three_center_general_caller<0, 5, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20105: three_center_general_caller<1, 5, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20205: three_center_general_caller<2, 5, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20305: three_center_general_caller<3, 5, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20405: three_center_general_caller<4, 5, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20505: three_center_general_caller<5, 5, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20006: three_center_general_caller<0, 6, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20106: three_center_general_caller<1, 6, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20206: three_center_general_caller<2, 6, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20306: three_center_general_caller<3, 6, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20406: three_center_general_caller<4, 6, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20506: three_center_general_caller<5, 6, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 20606: three_center_general_caller<6, 6, 2>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30000: three_center_general_caller<0, 0, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30001: three_center_general_caller<0, 1, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30101: three_center_general_caller<1, 1, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30002: three_center_general_caller<0, 2, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30102: three_center_general_caller<1, 2, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30202: three_center_general_caller<2, 2, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30003: three_center_general_caller<0, 3, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30103: three_center_general_caller<1, 3, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30203: three_center_general_caller<2, 3, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30303: three_center_general_caller<3, 3, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30004: three_center_general_caller<0, 4, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30104: three_center_general_caller<1, 4, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30204: three_center_general_caller<2, 4, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30304: three_center_general_caller<3, 4, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30404: three_center_general_caller<4, 4, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30005: three_center_general_caller<0, 5, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30105: three_center_general_caller<1, 5, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30205: three_center_general_caller<2, 5, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30305: three_center_general_caller<3, 5, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30405: three_center_general_caller<4, 5, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30505: three_center_general_caller<5, 5, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30006: three_center_general_caller<0, 6, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30106: three_center_general_caller<1, 6, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30206: three_center_general_caller<2, 6, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30306: three_center_general_caller<3, 6, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30406: three_center_general_caller<4, 6, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30506: three_center_general_caller<5, 6, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 30606: three_center_general_caller<6, 6, 3>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40000: three_center_general_caller<0, 0, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40001: three_center_general_caller<0, 1, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40101: three_center_general_caller<1, 1, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40002: three_center_general_caller<0, 2, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40102: three_center_general_caller<1, 2, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40202: three_center_general_caller<2, 2, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40003: three_center_general_caller<0, 3, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40103: three_center_general_caller<1, 3, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40203: three_center_general_caller<2, 3, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40303: three_center_general_caller<3, 3, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40004: three_center_general_caller<0, 4, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40104: three_center_general_caller<1, 4, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40204: three_center_general_caller<2, 4, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40304: three_center_general_caller<3, 4, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40404: three_center_general_caller<4, 4, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40005: three_center_general_caller<0, 5, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40105: three_center_general_caller<1, 5, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40205: three_center_general_caller<2, 5, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40305: three_center_general_caller<3, 5, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40405: three_center_general_caller<4, 5, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40505: three_center_general_caller<5, 5, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40006: three_center_general_caller<0, 6, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40106: three_center_general_caller<1, 6, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40206: three_center_general_caller<2, 6, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40306: three_center_general_caller<3, 6, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40406: three_center_general_caller<4, 6, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40506: three_center_general_caller<5, 6, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 40606: three_center_general_caller<6, 6, 4>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50000: three_center_general_caller<0, 0, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50001: three_center_general_caller<0, 1, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50101: three_center_general_caller<1, 1, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50002: three_center_general_caller<0, 2, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50102: three_center_general_caller<1, 2, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50202: three_center_general_caller<2, 2, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50003: three_center_general_caller<0, 3, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50103: three_center_general_caller<1, 3, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50203: three_center_general_caller<2, 3, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50303: three_center_general_caller<3, 3, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50004: three_center_general_caller<0, 4, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50104: three_center_general_caller<1, 4, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50204: three_center_general_caller<2, 4, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50304: three_center_general_caller<3, 4, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50404: three_center_general_caller<4, 4, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50005: three_center_general_caller<0, 5, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50105: three_center_general_caller<1, 5, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50205: three_center_general_caller<2, 5, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50305: three_center_general_caller<3, 5, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50405: three_center_general_caller<4, 5, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50505: three_center_general_caller<5, 5, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50006: three_center_general_caller<0, 6, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50106: three_center_general_caller<1, 6, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50206: three_center_general_caller<2, 6, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50306: three_center_general_caller<3, 6, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50406: three_center_general_caller<4, 6, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50506: three_center_general_caller<5, 6, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 50606: three_center_general_caller<6, 6, 5>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60000: three_center_general_caller<0, 0, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60001: three_center_general_caller<0, 1, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60101: three_center_general_caller<1, 1, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60002: three_center_general_caller<0, 2, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60102: three_center_general_caller<1, 2, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60202: three_center_general_caller<2, 2, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60003: three_center_general_caller<0, 3, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60103: three_center_general_caller<1, 3, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60203: three_center_general_caller<2, 3, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60303: three_center_general_caller<3, 3, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60004: three_center_general_caller<0, 4, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60104: three_center_general_caller<1, 4, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60204: three_center_general_caller<2, 4, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60304: three_center_general_caller<3, 4, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60404: three_center_general_caller<4, 4, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60005: three_center_general_caller<0, 5, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60105: three_center_general_caller<1, 5, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60205: three_center_general_caller<2, 5, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60305: three_center_general_caller<3, 5, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60405: three_center_general_caller<4, 5, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60505: three_center_general_caller<5, 5, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60006: three_center_general_caller<0, 6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60106: three_center_general_caller<1, 6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60206: three_center_general_caller<2, 6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60306: three_center_general_caller<3, 6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60406: three_center_general_caller<4, 6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60506: three_center_general_caller<5, 6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 60606: three_center_general_caller<6, 6, 6>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70000: three_center_general_caller<0, 0, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70001: three_center_general_caller<0, 1, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70101: three_center_general_caller<1, 1, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70002: three_center_general_caller<0, 2, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70102: three_center_general_caller<1, 2, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70202: three_center_general_caller<2, 2, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70003: three_center_general_caller<0, 3, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70103: three_center_general_caller<1, 3, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70203: three_center_general_caller<2, 3, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70303: three_center_general_caller<3, 3, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70004: three_center_general_caller<0, 4, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70104: three_center_general_caller<1, 4, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70204: three_center_general_caller<2, 4, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70304: three_center_general_caller<3, 4, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70404: three_center_general_caller<4, 4, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70005: three_center_general_caller<0, 5, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70105: three_center_general_caller<1, 5, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70205: three_center_general_caller<2, 5, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70305: three_center_general_caller<3, 5, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70405: three_center_general_caller<4, 5, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70505: three_center_general_caller<5, 5, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70006: three_center_general_caller<0, 6, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70106: three_center_general_caller<1, 6, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70206: three_center_general_caller<2, 6, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70306: three_center_general_caller<3, 6, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70406: three_center_general_caller<4, 6, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70506: three_center_general_caller<5, 6, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 70606: three_center_general_caller<6, 6, 7>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80000: three_center_general_caller<0, 0, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80001: three_center_general_caller<0, 1, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80101: three_center_general_caller<1, 1, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80002: three_center_general_caller<0, 2, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80102: three_center_general_caller<1, 2, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80202: three_center_general_caller<2, 2, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80003: three_center_general_caller<0, 3, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80103: three_center_general_caller<1, 3, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80203: three_center_general_caller<2, 3, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80303: three_center_general_caller<3, 3, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80004: three_center_general_caller<0, 4, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80104: three_center_general_caller<1, 4, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80204: three_center_general_caller<2, 4, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80304: three_center_general_caller<3, 4, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80404: three_center_general_caller<4, 4, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80005: three_center_general_caller<0, 5, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80105: three_center_general_caller<1, 5, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80205: three_center_general_caller<2, 5, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80305: three_center_general_caller<3, 5, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80405: three_center_general_caller<4, 5, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80505: three_center_general_caller<5, 5, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80006: three_center_general_caller<0, 6, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80106: three_center_general_caller<1, 6, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80206: three_center_general_caller<2, 6, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80306: three_center_general_caller<3, 6, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80406: three_center_general_caller<4, 6, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80506: three_center_general_caller<5, 6, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 80606: three_center_general_caller<6, 6, 8>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90000: three_center_general_caller<0, 0, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90001: three_center_general_caller<0, 1, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90101: three_center_general_caller<1, 1, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90002: three_center_general_caller<0, 2, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90102: three_center_general_caller<1, 2, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90202: three_center_general_caller<2, 2, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90003: three_center_general_caller<0, 3, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90103: three_center_general_caller<1, 3, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90203: three_center_general_caller<2, 3, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90303: three_center_general_caller<3, 3, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90004: three_center_general_caller<0, 4, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90104: three_center_general_caller<1, 4, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90204: three_center_general_caller<2, 4, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90304: three_center_general_caller<3, 4, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90404: three_center_general_caller<4, 4, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90005: three_center_general_caller<0, 5, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90105: three_center_general_caller<1, 5, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90205: three_center_general_caller<2, 5, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90305: three_center_general_caller<3, 5, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90405: three_center_general_caller<4, 5, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90505: three_center_general_caller<5, 5, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90006: three_center_general_caller<0, 6, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90106: three_center_general_caller<1, 6, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90206: three_center_general_caller<2, 6, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90306: three_center_general_caller<3, 6, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90406: three_center_general_caller<4, 6, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90506: three_center_general_caller<5, 6, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            case 90606: three_center_general_caller<6, 6, 9>(pair_P_p, pair_A_a, pair_B_b, pair_coefficient, pair_i_ao_start, pair_j_ao_start, n_pair_ij, aux_A_a, aux_coefficient, aux_i_aux_start, n_aux_primitive_k, J3c_matrix, n_ao, n_aux, spherical, omega); break;
            default:
                printf("%s function does not support angular i_L = %d, j_L = %d, k_L = %d\n", __func__ , i_L, j_L, k_L);
                fflush(stdout);
                return 1;
        }
        return 0;
    }
}
