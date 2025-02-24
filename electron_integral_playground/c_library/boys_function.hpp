
#pragma once

#include "angular.h"
#include "boys_function_interpolate_parameter.h"

#include <math.h>

static const double boys_infinity_prefactor[MAX_BOYS_ORDER + 1] {
    1.0/2.0*sqrt(M_PI),
    1.0/4.0*sqrt(M_PI),
    3.0/8.0*sqrt(M_PI),
    15.0/16.0*sqrt(M_PI),
    105.0/32.0*sqrt(M_PI),
    945.0/64.0*sqrt(M_PI),
    10395.0/128.0*sqrt(M_PI),
    135135.0/256.0*sqrt(M_PI),
    2027025.0/512.0*sqrt(M_PI),
    34459425.0/1024.0*sqrt(M_PI),
    654729075.0/2048.0*sqrt(M_PI),
    13749310575.0/4096.0*sqrt(M_PI),
    316234143225.0/8192.0*sqrt(M_PI),
    7905853580625.0/16384.0*sqrt(M_PI),
    213458046676875.0/32768.0*sqrt(M_PI),
    6190283353629375.0/65536.0*sqrt(M_PI),
    191898783962510625.0/131072.0*sqrt(M_PI),
    6332659870762850625.0/262144.0*sqrt(M_PI),
    221643095476699771875.0/524288.0*sqrt(M_PI),
    8200794532637891559375.0/1048576.0*sqrt(M_PI),
    319830986772877770815625.0/2097152.0*sqrt(M_PI),
    13113070457687988603440625.0/4194304.0*sqrt(M_PI),
    563862029680583509947946875.0/8388608.0*sqrt(M_PI),
    25373791335626257947657609375.0/16777216.0*sqrt(M_PI),
    1192568192774434123539907640625.0/33554432.0*sqrt(M_PI),
    58435841445947272053455474390625.0/67108864.0*sqrt(M_PI),
    2980227913743310874726229193921875.0/134217728.0*sqrt(M_PI),
};

#define BOYS_TAYLOR_ORDER 8

static const double boys_taylor_prefactor[MAX_BOYS_ORDER + 1][BOYS_TAYLOR_ORDER + 1] {
    {           1.0,      -1.0/3.0,      1.0/10.0,     -1.0/42.0,     1.0/216.0,   -1.0/1320.0,    1.0/9360.0,  -1.0/75600.0,  1.0/685440.0, },
    {       1.0/3.0,      -1.0/5.0,      1.0/14.0,     -1.0/54.0,     1.0/264.0,   -1.0/1560.0,   1.0/10800.0,  -1.0/85680.0,  1.0/766080.0, },
    {       1.0/5.0,      -1.0/7.0,      1.0/18.0,     -1.0/66.0,     1.0/312.0,   -1.0/1800.0,   1.0/12240.0,  -1.0/95760.0,  1.0/846720.0, },
    {       1.0/7.0,      -1.0/9.0,      1.0/22.0,     -1.0/78.0,     1.0/360.0,   -1.0/2040.0,   1.0/13680.0, -1.0/105840.0,  1.0/927360.0, },
    {       1.0/9.0,     -1.0/11.0,      1.0/26.0,     -1.0/90.0,     1.0/408.0,   -1.0/2280.0,   1.0/15120.0, -1.0/115920.0, 1.0/1008000.0, },
    {      1.0/11.0,     -1.0/13.0,      1.0/30.0,    -1.0/102.0,     1.0/456.0,   -1.0/2520.0,   1.0/16560.0, -1.0/126000.0, 1.0/1088640.0, },
    {      1.0/13.0,     -1.0/15.0,      1.0/34.0,    -1.0/114.0,     1.0/504.0,   -1.0/2760.0,   1.0/18000.0, -1.0/136080.0, 1.0/1169280.0, },
    {      1.0/15.0,     -1.0/17.0,      1.0/38.0,    -1.0/126.0,     1.0/552.0,   -1.0/3000.0,   1.0/19440.0, -1.0/146160.0, 1.0/1249920.0, },
    {      1.0/17.0,     -1.0/19.0,      1.0/42.0,    -1.0/138.0,     1.0/600.0,   -1.0/3240.0,   1.0/20880.0, -1.0/156240.0, 1.0/1330560.0, },
    {      1.0/19.0,     -1.0/21.0,      1.0/46.0,    -1.0/150.0,     1.0/648.0,   -1.0/3480.0,   1.0/22320.0, -1.0/166320.0, 1.0/1411200.0, },
    {      1.0/21.0,     -1.0/23.0,      1.0/50.0,    -1.0/162.0,     1.0/696.0,   -1.0/3720.0,   1.0/23760.0, -1.0/176400.0, 1.0/1491840.0, },
    {      1.0/23.0,     -1.0/25.0,      1.0/54.0,    -1.0/174.0,     1.0/744.0,   -1.0/3960.0,   1.0/25200.0, -1.0/186480.0, 1.0/1572480.0, },
    {      1.0/25.0,     -1.0/27.0,      1.0/58.0,    -1.0/186.0,     1.0/792.0,   -1.0/4200.0,   1.0/26640.0, -1.0/196560.0, 1.0/1653120.0, },
    {      1.0/27.0,     -1.0/29.0,      1.0/62.0,    -1.0/198.0,     1.0/840.0,   -1.0/4440.0,   1.0/28080.0, -1.0/206640.0, 1.0/1733760.0, },
    {      1.0/29.0,     -1.0/31.0,      1.0/66.0,    -1.0/210.0,     1.0/888.0,   -1.0/4680.0,   1.0/29520.0, -1.0/216720.0, 1.0/1814400.0, },
    {      1.0/31.0,     -1.0/33.0,      1.0/70.0,    -1.0/222.0,     1.0/936.0,   -1.0/4920.0,   1.0/30960.0, -1.0/226800.0, 1.0/1895040.0, },
    {      1.0/33.0,     -1.0/35.0,      1.0/74.0,    -1.0/234.0,     1.0/984.0,   -1.0/5160.0,   1.0/32400.0, -1.0/236880.0, 1.0/1975680.0, },
    {      1.0/35.0,     -1.0/37.0,      1.0/78.0,    -1.0/246.0,    1.0/1032.0,   -1.0/5400.0,   1.0/33840.0, -1.0/246960.0, 1.0/2056320.0, },
    {      1.0/37.0,     -1.0/39.0,      1.0/82.0,    -1.0/258.0,    1.0/1080.0,   -1.0/5640.0,   1.0/35280.0, -1.0/257040.0, 1.0/2136960.0, },
    {      1.0/39.0,     -1.0/41.0,      1.0/86.0,    -1.0/270.0,    1.0/1128.0,   -1.0/5880.0,   1.0/36720.0, -1.0/267120.0, 1.0/2217600.0, },
    {      1.0/41.0,     -1.0/43.0,      1.0/90.0,    -1.0/282.0,    1.0/1176.0,   -1.0/6120.0,   1.0/38160.0, -1.0/277200.0, 1.0/2298240.0, },
    {      1.0/43.0,     -1.0/45.0,      1.0/94.0,    -1.0/294.0,    1.0/1224.0,   -1.0/6360.0,   1.0/39600.0, -1.0/287280.0, 1.0/2378880.0, },
    {      1.0/45.0,     -1.0/47.0,      1.0/98.0,    -1.0/306.0,    1.0/1272.0,   -1.0/6600.0,   1.0/41040.0, -1.0/297360.0, 1.0/2459520.0, },
    {      1.0/47.0,     -1.0/49.0,     1.0/102.0,    -1.0/318.0,    1.0/1320.0,   -1.0/6840.0,   1.0/42480.0, -1.0/307440.0, 1.0/2540160.0, },
    {      1.0/49.0,     -1.0/51.0,     1.0/106.0,    -1.0/330.0,    1.0/1368.0,   -1.0/7080.0,   1.0/43920.0, -1.0/317520.0, 1.0/2620800.0, },
    {      1.0/51.0,     -1.0/53.0,     1.0/110.0,    -1.0/342.0,    1.0/1416.0,   -1.0/7320.0,   1.0/45360.0, -1.0/327600.0, 1.0/2701440.0, },
    {      1.0/53.0,     -1.0/55.0,     1.0/114.0,    -1.0/354.0,    1.0/1464.0,   -1.0/7560.0,   1.0/46800.0, -1.0/337680.0, 1.0/2782080.0, },
};

/*
    The following implementation of Boys function $F_m(x)$ guarantees that the max relative error is approximately $5 \times 10^{-14}$.

    The reference value is obtained according to the incomplete gamma function implementation of Boys function:
    $$F_m(x) = \frac{1}{2} \Gamma\left(m + \frac{1}{2}\right) \frac{1}{x^{m + 1/2}} \gamma\left(m + \frac{1}{2}, x\right)$$
    where
    $$\Gamma(n) = \int_0^\infty dt \ t^{n-1} e^{-t}$$
    is the Gamma function ($\Gamma\left(m + \frac{1}{2}\right) = \frac{(2m - 1)!!}{2^m} \sqrt{\pi}$), and
    $$\gamma(n,x) = \frac{1}{\Gamma(n)} \int_0^x dt \ t^{n-1} e^{-t}$$
    is the regularized lower incomplete gamma function. The implementation of regularized lower incomplete gamma function is obtained from scipy (\texttt{scipy.special.gammainc()}).

    In our implementation, the Boys function is evaluated in the following way:

    If $x = 0$, $F_m(0) = \frac{1}{2m + 1}$. We make this a special case because $x = 0$ is a frequently-encountered input.

    If $x \rightarrow 0$, the Taylor expansion of $F_m(x)$ at $x = 0$ is used:
    $$F_m(x) = \sum_{k=0}^\infty \frac{(-1)^k}{k!(2m+2k+1)} x^k$$
    In practice, an 8-th order Taylor expansion provides $5 \times 10^{-14}$ relative error for $F_{26}(x)$ at about $x \leq 0.1$.

    If $x \rightarrow \infty$, the asymptotic approximation of $F_m(x)$ is used:
    $$F_0(x) \approx \frac{1}{2} \sqrt{\frac{\pi}{x}}$$
    $$F_m(x) \approx \frac{(2m-1)!!}{2^{m+1}} \sqrt{\frac{\pi}{x^{2m+1}}} \quad (m \geq 1)$$
    In practice, this asymptotic approximation provides $5 \times 10^{-14}$ relative error for $F_{26}(x)$ at about $x \geq 85$.

    For the middle range of $F_m(x)$, the upward recursion is applied:
    $$F_0(x) = \frac{1}{2} \sqrt{\frac{\pi}{x}}erf(\sqrt{x})$$
    $$F_{m+1}(x) = \frac{(2m+1)F_m(x) - e^{-x}}{2x}$$
    In practice, the upward recursion is numerically unstable for small value of $x$, and provides $5 \times 10^{-14}$ relative error for $F_{26}(x)$ only at about $x \geq 18$.

    For $0.1 \leq x \leq 18$, a Chebyshev interpolation is used for every order ($m$) of $F_m(x)$. The interval of each spline is $0.1$ to ensure $5 \times 10^{-14}$ relative error for $F_{26}(x)$.
*/
template<int L> requires (L >= 0 && L <= MAX_BOYS_ORDER)
static void boys_function_evaluate(const double x, double y[L + 1])
{
    if (x < 1e-16)
    {
#pragma GCC ivdep
        for (int m = 0; m <= L; m++)
            y[m] = boys_taylor_prefactor[m][0];
    }
    else if (x < 0.1)
    {
        // Max relative error at L = 24, x = 0.01 is 4.4e-14.
        // A larger error is observed for x < 0.01, but I believe it's a problem of the reference value from scipy.
        double x_power[BOYS_TAYLOR_ORDER + 1];
        x_power[0] = 1.0;
        for (int k = 1; k <= BOYS_TAYLOR_ORDER; k++)
            x_power[k] = x_power[k - 1] * x;
#pragma GCC ivdep
        for (int m = 0; m <= L; m++)
        {
            double y_taylor = boys_taylor_prefactor[m][0];
            for (int k = 1; k <= BOYS_TAYLOR_ORDER; k++)
            {
                y_taylor += boys_taylor_prefactor[m][k] * x_power[k];
            }
            y[m] = y_taylor;
        }
    }
    else if (x < 18.0)
    {
        // Max relative error at L = 26, 0.1 <= x <= 18.0 is 3.2e-14.
        const int i_interval = (int) floor((x - boys_chebyshev_x0) * one_over_boys_chebyshev_x_interval);
        const double x_minus_x0 = x - (boys_chebyshev_x0 + i_interval * boys_chebyshev_x_interval);
        const double x_in_chebyshev_window = x_minus_x0 * one_over_boys_chebyshev_x_interval * 2.0 - 1.0;
        double x_power[BOYS_CHEBYSHEV_POLYNOMIAL_ORDER + 1];
        x_power[0] = 1.0;
        for (int k = 1; k <= BOYS_CHEBYSHEV_POLYNOMIAL_ORDER; k++)
            x_power[k] = x_power[k - 1] * x_in_chebyshev_window;
#pragma GCC ivdep
        for (int m = 0; m <= L; m++)
        {
            double y_chebyshev = boys_chebyshev_coefficient[m][i_interval][0];
            for (int k = 1; k <= BOYS_CHEBYSHEV_POLYNOMIAL_ORDER; k++)
            {
                y_chebyshev += boys_chebyshev_coefficient[m][i_interval][k] * x_power[k];
            }
            y[m] = y_chebyshev;
        }
    }
    else if (x < 85.0)
    {
        // Max relative error at L = 26, x = 18.0 is 4.4e-14.
        const double sqrt_x = sqrt(x);
        double boys_m = boys_infinity_prefactor[0] * erf(sqrt_x) / sqrt_x;
        y[0] = boys_m;
        if constexpr (L == 0)
            return;
        const double exp_minus_x = exp(-x);
        const double one_over_two_x = 0.5 / x;
        for (int m = 1; m <= L; m++)
        {
            boys_m = ((2 * m - 1) * boys_m - exp_minus_x) * one_over_two_x;
            y[m] = boys_m;
        }
    }
    else
    {
        // max relative error at L = 26, x = 85.0 is 3.5e-14
#pragma GCC ivdep
        for (int m = 0; m <= L; m++)
        {
            y[m] = boys_infinity_prefactor[m] * pow(x, -m - 0.5);
        }
    }
}


