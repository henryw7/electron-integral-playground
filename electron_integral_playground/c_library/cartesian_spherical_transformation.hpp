
#include <math.h>

template<int L, int increment>
class CartesianToSpherical
{
public:
    static void apply(double* vector);
};

template<int increment>
class CartesianToSpherical<0, increment>
{
public:
    static void apply(double* vector) {}
};

template<int increment>
class CartesianToSpherical<1, increment>
{
public:
    static void apply(double* vector) {}
};

template<int increment>
class CartesianToSpherical<2, increment>
{
public:
    static void apply(double* vector)
    {
        const double spherical_z2 = vector[5 * increment] - 0.5 * (vector[0 * increment] + vector[3 * increment]);
        const double spherical_x2_y2 = 0.5 * sqrt(3.0) * (vector[0 * increment] - vector[3 * increment]);
        vector[0 * increment] = vector[1 * increment]; // xy
        vector[1 * increment] = vector[4 * increment]; // yz
        vector[3 * increment] = vector[2 * increment]; // xz
        vector[2 * increment] = spherical_z2;
        vector[4 * increment] = spherical_x2_y2;
        vector[5 * increment] = NAN;
    }
};

template<int i_L, int j_L>
static void cartesian_to_spherical(double matrix[(i_L + 1) * (i_L + 2) / 2 * (j_L + 1) * (j_L + 2) / 2])
{
    constexpr int n_cartesian_i = (i_L + 1) * (i_L + 2) / 2;
    constexpr int n_cartesian_j = (j_L + 1) * (j_L + 2) / 2;
    for (int i = 0; i < n_cartesian_i; i++)
    {
        CartesianToSpherical<j_L, 1>::apply(matrix + i * n_cartesian_j);
    }
    constexpr int n_spherical_j = j_L * 2 + 1;
    for (int j = 0; j < n_spherical_j; j++)
    {
        CartesianToSpherical<i_L, n_cartesian_j>::apply(matrix + j);
    }
}

