
#pragma once

#include "angular.h"

#include <math.h>

/*
    The function form of spherical Gaussian orbitals can be generated with the following generator functions:
    $$\gamma_{l,k,m} = (-1)^k 2^{-l} \left({\begin{array}{*{20}c} l \\ k \end{array}}\right) \left({\begin{array}{*{20}c} 2l - 2k \\ l \end{array}}\right) \frac{(l - 2k)!}{(l - 2k - m)!}$$
    $$\Pi_{l,m}(z, r) = \sum_{k = 0}^{floor((l - m) / 2)} \gamma_{l,k,m} r^{2k} z^{l -2k - m}$$
    $$A_m(x, y) = \sum_{p = 0}^m \left({\begin{array}{*{20}c} m \\ p \end{array}}\right) x^p y^{m - p} cos\left(\frac{\pi}{2} (m - p)\right)$$
    $$B_m(x, y) = \sum_{p = 0}^m \left({\begin{array}{*{20}c} m \\ p \end{array}}\right) x^p y^{m - p} sin\left(\frac{\pi}{2} (m - p)\right)$$
    $$C_{l, m}(x, y, z) = \sqrt{\frac{(2 - \delta_{m, 0}) (l - m)!}{(l + m)!}} \Pi_{l,m}(z, r) A_m(x, y) \qquad m = 0,1,...,l$$
    $$S_{l, m}(x, y, z) = \sqrt{\frac{2(l - m)!}{(l + m)!}} \Pi_{l,m}(z, r) B_m(x, y) \qquad m = 1,2,...,l$$
    The spherical Gaussian orbitals with the same $l$ are ordered by
    $$S_{l, l}, S_{l, l - 1}, ..., S_{l, 1}, C_{l, 0}, C_{l, 1}, ..., C_{l, l - 1}, C_{l, l}$$
*/

template<int L, int increment> requires (L >= 0 && L <= MAX_L)
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

// Note: the correct spherical Gaussian orbital order for p orbital is y,z,x.
//       But clearly nobody is using this order, so we do not apply this rotation as well.
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
        const double spherical_0 = -1.5/sqrt(5.0) * (vector[2 * increment] + vector[7 * increment]) + vector[9 * increment];
        const double spherical_1 = -0.5*sqrt(1.5) * vector[0 * increment] - 0.5*sqrt(0.3) * vector[3 * increment] + 2.0*sqrt(0.3) * vector[5 * increment];
        const double spherical_2 = sqrt(3.0)/2.0 * (vector[2 * increment] - vector[7 * increment]);
        const double spherical_3 = 0.5*sqrt(2.5) * vector[0 * increment] - 1.5/sqrt(2.0) * vector[3 * increment];
        vector[1 * increment] = vector[4 * increment]; // xyz
        vector[0 * increment] = spherical_minus_3;
        vector[2 * increment] = spherical_minus_1;
        vector[3 * increment] = spherical_0;
        vector[4 * increment] = spherical_1;
        vector[5 * increment] = spherical_2;
        vector[6 * increment] = spherical_3;
        vector[7 * increment] = NAN;
        vector[8 * increment] = NAN;
        vector[9 * increment] = NAN;
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
#pragma unroll
        for (int i = 9; i < 15; i++)
            vector[i * increment] = NAN;
    }
};

template<int increment>
class CartesianToSpherical<5, increment>
{
    const double C_transform[11 * 21]
    { //              x^5                x^4y                x^4z              x^3y^2               x^3yz              x^3z^2              x^2y^3             x^2y^2z             x^2yz^2              x^2z^3                xy^4               xy^3z             xy^2z^2               xyz^3                xz^4                 y^5                y^4z              y^3z^2              y^2z^3                yz^4                 z^5
                        0,  sqrt(175.0/128.0),                  0,                  0,                  0,                  0,   -sqrt(75.0/32.0),                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,   sqrt(63.0/128.0),                  0,                  0,                  0,                  0,                  0,
                        0,                  0,                  0,                  0,      sqrt(5.0/4.0),                  0,                  0,                  0,                  0,                  0,                  0,     -sqrt(5.0/4.0),                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,
                        0,  -sqrt(35.0/128.0),                  0,                  0,                  0,                  0,    -sqrt(5.0/96.0),                  0,      sqrt(3.0/2.0),                  0,                  0,                  0,                  0,                  0,                  0,   sqrt(35.0/128.0),                  0,     -sqrt(5.0/6.0),                  0,                  0,                  0,
                        0,                  0,                  0,                  0,    -sqrt(5.0/12.0),                  0,                  0,                  0,                  0,                  0,                  0,    -sqrt(5.0/12.0),                  0,      sqrt(5.0/3.0),                  0,                  0,                  0,                  0,                  0,                  0,                  0,
                        0,    sqrt(5.0/192.0),                  0,                  0,                  0,                  0,    sqrt(5.0/112.0),                  0,    -sqrt(9.0/28.0),                  0,                  0,                  0,                  0,                  0,                  0,    sqrt(15.0/64.0),                  0,   -sqrt(45.0/28.0),                  0,      sqrt(5.0/3.0),                  0,
                        0,                  0,            5.0/8.0,                  0,                  0,                  0,                  0,   sqrt(15.0/112.0),                  0,   -sqrt(25.0/21.0),                  0,                  0,                  0,                  0,                  0,                  0,            5.0/8.0,                  0,   -sqrt(25.0/21.0),                  0,                1.0,
          sqrt(15.0/64.0),                  0,                  0,    sqrt(5.0/112.0),                  0,   -sqrt(45.0/28.0),                  0,                  0,                  0,                  0,    sqrt(5.0/192.0),                  0,    -sqrt(9.0/28.0),                  0,      sqrt(5.0/3.0),                  0,                  0,                  0,                  0,                  0,                  0,
                        0,                  0,   -sqrt(35.0/48.0),                  0,                  0,                  0,                  0,                  0,                  0,      sqrt(5.0/4.0),                  0,                  0,                  0,                  0,                  0,                  0,    sqrt(35.0/48.0),                  0,     -sqrt(5.0/4.0),                  0,                  0,
        -sqrt(35.0/128.0),                  0,                  0,     sqrt(5.0/96.0),                  0,      sqrt(5.0/6.0),                  0,                  0,                  0,                  0,   sqrt(35.0/128.0),                  0,     -sqrt(3.0/2.0),                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,
                        0,                  0,    sqrt(35.0/64.0),                  0,                  0,                  0,                  0,   -sqrt(27.0/16.0),                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,    sqrt(35.0/64.0),                  0,                  0,                  0,                  0,
         sqrt(63.0/128.0),                  0,                  0,   -sqrt(75.0/32.0),                  0,                  0,                  0,                  0,                  0,                  0,  sqrt(175.0/128.0),                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,                  0,
    };

public:
    static void apply(double* vector)
    {
        const double spherical[11] {
            sqrt(175.0/128.0) * vector[1 * increment] - sqrt(75.0/32.0) * vector[6 * increment] + sqrt(63.0/128.0) * vector[15 * increment],
            sqrt(5.0/4.0) * (vector[4 * increment] - vector[11 * increment]),
            sqrt(35.0/128.0) * (-vector[1 * increment] + vector[15 * increment]) - sqrt(5.0/96.0) * vector[6 * increment] + sqrt(3.0/2.0) * vector[8 * increment] - sqrt(5.0/6.0) * vector[17 * increment],
            -sqrt(5.0/12.0) * (vector[4 * increment] + vector[11 * increment]) + sqrt(5.0/3.0)*vector[13 * increment],
            sqrt(5.0/192.0) * vector[1 * increment] + sqrt(5.0/112.0) * vector[6 * increment] - sqrt(9.0/28.0) * vector[8 * increment] + sqrt(15.0/64.0) * vector[15 * increment] - sqrt(45.0/28.0) * vector[17 * increment] + sqrt(5.0/3.0) * vector[19 * increment],
            5.0/8.0 * (vector[2 * increment] + vector[16 * increment]) + sqrt(15.0/112.0) * vector[7 * increment] - sqrt(25.0/21.0) * (vector[9 * increment] + vector[18 * increment]) + vector[20 * increment],
            sqrt(15.0/64.0) * vector[0 * increment] + sqrt(5.0/112.0) * vector[3 * increment] - sqrt(45.0/28.0) * vector[5 * increment] + sqrt(5.0/192.0) * vector[10 * increment] - sqrt(9.0/28.0) * vector[12 * increment] + sqrt(5.0/3.0) * vector[14 * increment],
            sqrt(35.0/48.0) * (-vector[2 * increment] + vector[16 * increment]) + sqrt(5.0/4.0) * (vector[9 * increment] - vector[18 * increment]),
            sqrt(35.0/128.0) * (-vector[0 * increment] + vector[10 * increment]) + sqrt(5.0/96.0) * vector[3 * increment] + sqrt(5.0/6.0) * vector[5 * increment] - sqrt(3.0/2.0) * vector[12 * increment],
            sqrt(35.0/64.0) * (vector[2 * increment] + vector[16 * increment]) - sqrt(27.0/16.0) * vector[7 * increment],
            sqrt(63.0/128.0) * vector[0 * increment] - sqrt(75.0/32.0) * vector[3 * increment] + sqrt(175.0/128.0) * vector[10 * increment],
        };
#pragma unroll
        for (int i = 0; i < 11; i++)
            vector[i * increment] = spherical[i];
#pragma unroll
        for (int i = 11; i < 21; i++)
            vector[i * increment] = NAN;
    }
};

template<int increment>
class CartesianToSpherical<6, increment>
{
    const double C_transform[13 * 28]
    { //              x^6                  x^5y                  x^5z                x^4y^2                 x^4yz                x^4z^2                x^3y^3               x^3y^2z               x^3yz^2                x^3z^3                x^2y^4               x^2y^3z             x^2y^2z^2               x^2yz^3                x^2z^4                  xy^5                 xy^4z               xy^3z^2               xy^2z^3                 xyz^4                  xz^5                   y^6                  y^5z                y^4z^2                y^3z^3                y^2z^4                  yz^5                   z^6
                        0,    sqrt(189.0/128.0),                    0,                    0,                    0,                    0,    -sqrt(125.0/32.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,    sqrt(189.0/128.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,
                        0,                    0,                    0,                    0,    sqrt(175.0/128.0),                    0,                    0,                    0,                    0,                    0,                    0,     -sqrt(75.0/32.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,     sqrt(63.0/128.0),                    0,                    0,                    0,                    0,                    0,
                        0,    -sqrt(63.0/176.0),                    0,                    0,                    0,                    0,                    0,                    0,      sqrt(75.0/44.0),                    0,                    0,                    0,                    0,                    0,                    0,     sqrt(63.0/176.0),                    0,     -sqrt(75.0/44.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,
                        0,                    0,                    0,                    0,  -sqrt(945.0/1408.0),                    0,                    0,                    0,                    0,                    0,                    0,    -sqrt(45.0/352.0),                    0,      sqrt(45.0/22.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,   sqrt(945.0/1408.0),                    0,     -sqrt(25.0/22.0),                    0,                    0,                    0,
                        0,   sqrt(105.0/1408.0),                    0,                    0,                    0,                    0,     sqrt(25.0/352.0),                    0,     -sqrt(10.0/11.0),                    0,                    0,                    0,                    0,                    0,                    0,   sqrt(105.0/1408.0),                    0,     -sqrt(10.0/11.0),                    0,      sqrt(70.0/33.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,
                        0,                    0,                    0,                    0,   sqrt(175.0/2112.0),                    0,                    0,                    0,                    0,                    0,                    0,     sqrt(25.0/176.0),                    0,     -sqrt(25.0/44.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,    sqrt(525.0/704.0),                    0,    -sqrt(125.0/44.0),                    0,      sqrt(21.0/11.0),                    0,
                -5.0/16.0,                    0,                    0,   -sqrt(75.0/2816.0),                    0,    sqrt(675.0/704.0),                    0,                    0,                    0,                    0,   -sqrt(75.0/2816.0),                    0,   sqrt(405.0/1232.0),                    0,     -sqrt(75.0/44.0),                    0,                    0,                    0,                    0,                    0,                    0,            -5.0/16.0,                    0,    sqrt(675.0/704.0),                    0,     -sqrt(75.0/44.0),                    0,                  1.0,
                        0,                    0,    sqrt(525.0/704.0),                    0,                    0,                    0,                    0,     sqrt(25.0/176.0),                    0,    -sqrt(125.0/44.0),                    0,                    0,                    0,                    0,                    0,                    0,   sqrt(175.0/2112.0),                    0,     -sqrt(25.0/44.0),                    0,      sqrt(21.0/11.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,
        sqrt(105.0/512.0),                    0,                    0,    sqrt(35.0/5632.0),                    0,     -sqrt(35.0/22.0),                    0,                    0,                    0,                    0,   -sqrt(35.0/5632.0),                    0,                    0,                    0,      sqrt(35.0/22.0),                    0,                    0,                    0,                    0,                    0,                    0,   -sqrt(105.0/512.0),                    0,      sqrt(35.0/22.0),                    0,     -sqrt(35.0/22.0),                    0,                    0,
                        0,                    0,  -sqrt(945.0/1408.0),                    0,                    0,                    0,                    0,     sqrt(45.0/352.0),                    0,      sqrt(25.0/22.0),                    0,                    0,                    0,                    0,                    0,                    0,   sqrt(945.0/1408.0),                    0,     -sqrt(45.0/22.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,
        -sqrt(63.0/256.0),                    0,                    0,   sqrt(525.0/2816.0),                    0,    sqrt(525.0/704.0),                    0,                    0,                    0,                    0,   sqrt(525.0/2816.0),                    0,   -sqrt(405.0/176.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,    -sqrt(63.0/256.0),                    0,    sqrt(525.0/704.0),                    0,                    0,                    0,                    0,
                        0,                    0,     sqrt(63.0/128.0),                    0,                    0,                    0,                    0,     -sqrt(75.0/32.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,    sqrt(175.0/128.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,
        sqrt(231.0/512.0),                    0,                    0,  -sqrt(1575.0/512.0),                    0,                    0,                    0,                    0,                    0,                    0,   sqrt(1575.0/512.0),                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,                    0,   -sqrt(231.0/512.0),                    0,                    0,                    0,                    0,                    0,                    0,
    };

public:
    static void apply(double* vector)
    {
        const double spherical[13] {
            sqrt(189.0/128.0) * (vector[1 * increment] + vector[15 * increment]) - sqrt(125.0/32.0) * vector[6 * increment],
            sqrt(175.0/128.0) * vector[4 * increment] - sqrt(75.0/32.0) * vector[11 * increment] + sqrt(63.0/128.0) * vector[22 * increment],
            sqrt(63.0/176.0) * (-vector[1 * increment] + vector[15 * increment]) + sqrt(75.0/44.0) * (vector[8 * increment] - vector[17 * increment]),
            sqrt(945.0/1408.0) * (-vector[4 * increment] + vector[22 * increment]) - sqrt(45.0/352.0) * vector[11 * increment] + sqrt(45.0/22.0) * vector[13 * increment] - sqrt(25.0/22.0) * vector[24 * increment],
            sqrt(105.0/1408.0) * (vector[1 * increment] + vector[15 * increment]) + sqrt(25.0/352.0) * vector[6 * increment] - sqrt(10.0/11.0) * (vector[8 * increment] + vector[17 * increment]) + sqrt(70.0/33.0) * vector[19 * increment],
            sqrt(175.0/2112.0) * vector[4 * increment] + sqrt(25.0/176.0) * vector[11 * increment] - sqrt(25.0/44.0) * vector[13 * increment] + sqrt(525.0/704.0) * vector[22 * increment] - sqrt(125.0/44.0) * vector[24 * increment] + sqrt(21.0/11.0) * vector[26 * increment],
            -5.0/16.0 * (vector[0 * increment] + vector[21 * increment]) - sqrt(75.0/2816.0) * (vector[3 * increment] + vector[10 * increment]) + sqrt(675.0/704.0) * (vector[5 * increment] + vector[23 * increment]) + sqrt(405.0/1232.0) * vector[12 * increment] - sqrt(75.0/44.0) * (vector[14 * increment] + vector[25 * increment]) + vector[27 * increment],
            sqrt(525.0/704.0) * vector[2 * increment] + sqrt(25.0/176.0) * vector[7 * increment] - sqrt(125.0/44.0) * vector[9 * increment] + sqrt(175.0/2112.0) * vector[16 * increment] - sqrt(25.0/44.0) * vector[18 * increment] + sqrt(21.0/11.0) * vector[20 * increment],
            sqrt(105.0/512.0) * (vector[0 * increment] - vector[21 * increment]) + sqrt(35.0/5632.0) * (vector[3 * increment] - vector[10 * increment]) + sqrt(35.0/22.0) * (-vector[5 * increment] + vector[14 * increment] + vector[23 * increment] - vector[25 * increment]),
            sqrt(945.0/1408.0) * (-vector[2 * increment] + vector[16 * increment]) + sqrt(45.0/352.0) * vector[7 * increment] + sqrt(25.0/22.0) * vector[9 * increment] - sqrt(45.0/22.0) * vector[18 * increment],
            -sqrt(63.0/256.0) * (vector[0 * increment] + vector[21 * increment]) + sqrt(525.0/2816.0) * (vector[3 * increment] + vector[10 * increment]) + sqrt(525.0/704.0) * (vector[5 * increment] + vector[23 * increment]) - sqrt(405.0/176.0) * vector[12 * increment],
            sqrt(63.0/128.0) * vector[2 * increment] - sqrt(75.0/32.0) * vector[7 * increment] + sqrt(175.0/128.0) * vector[16 * increment],
            sqrt(231.0/512.0) * (vector[0 * increment] - vector[21 * increment]) + sqrt(1575.0/512.0) * (-vector[3 * increment] + vector[10 * increment]),
        };
#pragma unroll
        for (int i = 0; i < 13; i++)
            vector[i * increment] = spherical[i];
#pragma unroll
        for (int i = 13; i < 28; i++)
            vector[i * increment] = NAN;
    }
};

template<int i_L, int j_L> requires (i_L >= 0 && i_L <= MAX_L && j_L >= 0 && j_L <= MAX_L)
static void cartesian_to_spherical_inplace(double matrix[(i_L + 1) * (i_L + 2) / 2 * (j_L + 1) * (j_L + 2) / 2])
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
