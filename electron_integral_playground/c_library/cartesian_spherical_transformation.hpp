
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
    const double C_transform[5 * 6]
    { //          x^2   xy   xz             y^2   yz            z^2
                    0, 1.0,   0,              0,   0,             0, // xy
                    0,   0,   0,              0, 1.0,             0, // yz
                 -0.5,   0,   0,           -0.5,   0,           1.0, // z^2
                    0,   0, 1.0,              0,   0,             0, // xz
        sqrt(3.0)/2.0,   0,   0, -sqrt(3.0)/2.0,   0,             0, // x^2-y^2s
    };
    const double C_extra[1 * 6]
    {
        1.0/sqrt(5.0),   0,   0,  1.0/sqrt(5.0),   0, 1.0/sqrt(5.0), // s
    };

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

template<int increment>
class CartesianToSpherical<3, increment>
{
    const double C_transform[7 * 10]
    { //           x^3            x^2y            x^2z            xy^2  xyz            xz^2             y^3            y^2z            yz^2            z^3
                     0,  1.5/sqrt(2.0),              0,              0,   0,              0, -0.5*sqrt(2.5),              0,              0,             0, // y(3x^2-y^2)
                     0,              0,              0,              0, 1.0,              0,              0,              0,              0,             0, // xyz
                     0, -0.5*sqrt(0.3),              0,              0,   0,              0, -0.5*sqrt(1.5),              0,  2.0*sqrt(0.3),             0, // yz^2
                     0,              0, -1.5/sqrt(5.0),              0,   0,              0,              0, -1.5/sqrt(5.0),              0,           1.0, // z^3
        -0.5*sqrt(1.5),              0,              0, -0.5*sqrt(0.3),   0,  2.0*sqrt(0.3),              0,              0,              0,             0, // xz^2
                     0,              0,  sqrt(3.0)/2.0,              0,   0,              0,              0, -sqrt(3.0)/2.0,              0,             0, // z(x^2-y^2)
         0.5*sqrt(2.5),              0,              0, -1.5/sqrt(2.0),   0,              0,              0,              0,              0,             0, // x(x^2-3y^2)
    };
    const double C_extra[3 * 10]
    {
         sqrt(3.0/7.0),              0,              0, sqrt(3.0/35.0),   0, sqrt(3.0/35.0),              0,              0,              0,             0, // px
                     0, sqrt(3.0/35.0),              0,              0,   0,              0,  sqrt(3.0/7.0),              0, sqrt(3.0/35.0),             0, // py
                     0,              0, sqrt(3.0/35.0),              0,   0,              0,              0, sqrt(3.0/35.0),              0, sqrt(3.0/7.0), // pz
    };

public:
    static void apply(double* vector)
    {
        const double f_minus_3 = 1.5/sqrt(2.0) * vector[1 * increment] - 0.5*sqrt(2.5) * vector[6 * increment];
        const double f_minus_1 = -0.5*sqrt(0.3) * vector[1 * increment] - 0.5*sqrt(1.5) * vector[6 * increment] + 2.0*sqrt(0.3) * vector[8 * increment];
        const double f_0 = -1.5/sqrt(5.0) * vector[2 * increment] - 1.5/sqrt(5.0) * vector[7 * increment] + vector[9 * increment];
        const double f_1 = -0.5*sqrt(1.5) * vector[0 * increment] - 0.5*sqrt(0.3) * vector[3 * increment] + 2.0*sqrt(0.3) * vector[5 * increment];
        const double f_2 = sqrt(3.0)/2.0 * vector[2 * increment] - sqrt(3.0)/2.0 * vector[7 * increment];
        const double f_3 = 0.5*sqrt(2.5) * vector[0 * increment] - 1.5/sqrt(2.0) * vector[3 * increment];
        vector[1 * increment] = vector[4 * increment]; // xyz
        vector[0 * increment] = f_minus_3;
        vector[2 * increment] = f_minus_1;
        vector[3 * increment] = f_0;
        vector[4 * increment] = f_1;
        vector[5 * increment] = f_2;
        vector[6 * increment] = f_3;
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
