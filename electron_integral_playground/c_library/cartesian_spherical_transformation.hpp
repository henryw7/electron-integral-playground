
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
        const double spherical_minus_3 = 1.5/sqrt(2.0) * vector[1 * increment] - 0.5*sqrt(2.5) * vector[6 * increment];
        const double spherical_minus_1 = -0.5*sqrt(0.3) * vector[1 * increment] - 0.5*sqrt(1.5) * vector[6 * increment] + 2.0*sqrt(0.3) * vector[8 * increment];
        const double spherical_0 = -1.5/sqrt(5.0) * vector[2 * increment] - 1.5/sqrt(5.0) * vector[7 * increment] + vector[9 * increment];
        const double spherical_1 = -0.5*sqrt(1.5) * vector[0 * increment] - 0.5*sqrt(0.3) * vector[3 * increment] + 2.0*sqrt(0.3) * vector[5 * increment];
        const double spherical_2 = sqrt(3.0)/2.0 * vector[2 * increment] - sqrt(3.0)/2.0 * vector[7 * increment];
        const double spherical_3 = 0.5*sqrt(2.5) * vector[0 * increment] - 1.5/sqrt(2.0) * vector[3 * increment];
        vector[1 * increment] = vector[4 * increment]; // xyz
        vector[0 * increment] = spherical_minus_3;
        vector[2 * increment] = spherical_minus_1;
        vector[3 * increment] = spherical_0;
        vector[4 * increment] = spherical_1;
        vector[5 * increment] = spherical_2;
        vector[6 * increment] = spherical_3;
    }
};

template<int increment>
class CartesianToSpherical<4, increment>
{
    const double C_transform[9 * 15]
    { //            x^4              x^3y              x^3z            x^2y^2             x^2yz            x^2z^2              xy^3             xy^2z             xyz^2              xz^3               y^4              y^3z            y^2z^2              yz^3               z^4
                      0,    sqrt(5.0/4.0),                0,                0,                0,                0,   -sqrt(5.0/4.0),                0,                0,                0,                0,                0,                0,                0,                0,
                      0,                0,                0,                0,    sqrt(9.0/8.0),                0,                0,                0,                0,                0,                0,   -sqrt(5.0/8.0),                0,                0,                0,
                      0,  -sqrt(5.0/28.0),                0,                0,                0,                0,  -sqrt(5.0/28.0),                0,    sqrt(9.0/7.0),                0,                0,                0,                0,                0,                0,
                      0,                0,                0,                0,  -sqrt(9.0/56.0),                0,                0,                0,                0,                0,                0, -sqrt(45.0/56.0),                0,   sqrt(10.0/7.0),                0,
                3.0/8.0,                0,                0, sqrt(27.0/560.0),                0, -sqrt(27.0/35.0),                0,                0,                0,                0,          3.0/8.0,                0, -sqrt(27.0/35.0),                0,              1.0,
                      0,                0, -sqrt(45.0/56.0),                0,                0,                0,                0,  -sqrt(9.0/56.0),                0,   sqrt(10.0/7.0),                0,                0,                0,                0,                0,
        -sqrt(5.0/16.0),                0,                0,                0,                0,  sqrt(27.0/28.0),                0,                0,                0,                0,   sqrt(5.0/16.0),                0, -sqrt(27.0/28.0),                0,                0,
                      0,                0,    sqrt(5.0/8.0),                0,                0,                0,                0,   -sqrt(9.0/8.0),                0,                0,                0,                0,                0,                0,                0,
        sqrt(35.0/64.0),                0,                0, -sqrt(27.0/16.0),                0,                0,                0,                0,                0,                0,  sqrt(35.0/64.0),                0,                0,                0,                0,
    };

public:
    static void apply(double* vector)
    {
        const double spherical[9] {
            sqrt(5.0/4.0) * (vector[1 * increment] - vector[6 * increment]),
            sqrt(9.0/8.0) * vector[4 * increment] - sqrt(5.0/8.0) * vector[11 * increment],
            -sqrt(5.0/28.0) * (vector[1 * increment] + vector[6 * increment]) + sqrt(9.0/7.0) * vector[8 * increment],
            -sqrt(9.0/56.0) * vector[4 * increment] - sqrt(45.0/56.0) * vector[11 * increment] + sqrt(10.0/7.0) * vector[13 * increment],
            3.0/8.0 * (vector[0 * increment] + vector[10 * increment]) + sqrt(27.0/560.0) * vector[3 * increment] - sqrt(27.0/35.0) * (vector[5 * increment] + vector[12 * increment]) + vector[14 * increment],
            -sqrt(45.0/56.0) * vector[2 * increment] - sqrt(9.0/56.0) * vector[7 * increment] + sqrt(10.0/7.0) * vector[9 * increment],
            sqrt(5.0/16.0) * (-vector[0 * increment] + vector[10 * increment]) + sqrt(27.0/28.0) * (vector[5 * increment] - vector[12 * increment]),
            sqrt(5.0/8.0) * vector[2 * increment] - sqrt(9.0/8.0) * vector[7 * increment],
            sqrt(35.0/64.0) * (vector[0 * increment] + vector[10 * increment]) - sqrt(27.0/16.0) * vector[3 * increment],
        };
#pragma unroll
        for (int i = 0; i < 9; i++)
            vector[i * increment] = spherical[i];
    }
};

template<int i_L, int j_L>
static void cartesian_to_spherical(double matrix[(i_L + 1) * (i_L + 2) / 2 * (j_L + 1) * (j_L + 2) / 2])
{
    constexpr int n_cartesian_i = (i_L + 1) * (i_L + 2) / 2;
    constexpr int n_cartesian_j = (j_L + 1) * (j_L + 2) / 2;
#pragma unroll
    for (int i = 0; i < n_cartesian_i; i++)
    {
        CartesianToSpherical<j_L, 1>::apply(matrix + i * n_cartesian_j);
    }
    constexpr int n_spherical_j = j_L * 2 + 1;
#pragma unroll
    for (int j = 0; j < n_spherical_j; j++)
    {
        CartesianToSpherical<i_L, n_cartesian_j>::apply(matrix + j);
    }
}
