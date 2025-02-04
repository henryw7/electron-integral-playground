
#pragma once

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
    Consistent with the following for loop:
    int index = 0;
    for (int row = 0; row <= L; row++)
        for (int column = 0; column <= row; column++)
            index++;
*/
static inline int lower_triangular_index(const int row, const int column)
{
    return row * (row + 1) / 2 + column;
}

template <int L> requires (L >= 0)
constexpr int lower_triangular_total = (L + 1) * (L + 2) / 2;

/*
    Consistent with the following for loop:
    int index = 0;
    for (int row = 0; row <= L; row++)
        for (int column = 0; column <= row; column++)
        {
            if ((row + column) % 2 != 0)
                continue;
            index++;
        }
*/
static inline int lower_triangular_even_index(const int row, const int column)
{
    // Attention: flooring function is taken by integer division
    return (row / 2) * (row / 2 + 1) + (row % 2) * (row / 2 + 1) + column / 2;
}

template <int L> requires (L >= 0)
// Attention: flooring function is taken by integer division
constexpr int lower_triangular_even_total = ((L + 1) / 2 + 1) * (L / 2 + 1);

/*
    Consistent with the following for loop:
    int index = 0;
    for (int row = 0; row <= L; row++)
        for (int column = 0; column <= min(row, L - row); column++)
            index++;
*/
template <int L> requires (L >= 0)
static inline int lower_triangular_upper_anti_triangular_index(const int row, const int column)
{
    // Attention: flooring function is taken by integer division
    const int half_L = (L + 1) / 2;
    if (row <= half_L)
        return row * (row + 1) / 2 + column;
    else
        return half_L * (half_L + 1) / 2 + ((L + 2) / 2 + L + 2 - row) * (row - half_L) / 2 + column;
}

template <int L> requires (L >= 0)
// Attention: flooring function is taken by integer division
constexpr int lower_triangular_upper_anti_triangular_total = ((L + 1) / 2 + 1) * (L / 2 + 1);

/*
    Consistent with the following for loop:
    int index = 0;
    for (int i = 0; i <= i_L; i++)
        for (int j = 0; j <= j_L; j++)
            for (int t = 0; t <= i + j; t++)
                index++;
*/
template <int j_L> requires (j_L >= 0)
static inline int mcmurchie_davidson_E_ijt_index(const int i, const int j, const int t)
{
    return i * (j_L + 1) * (j_L + 2) / 2 + (i - 1) * i / 2 * (j_L + 1) + j * (j + 2 * i + 1) / 2 + t;
}

template <int i_L, int j_L> requires (i_L >= 0 && j_L >= 0)
constexpr int mcmurchie_davidson_E_ijt_total = (i_L + 1) * (j_L + 1) * (j_L + 2) / 2 + (j_L + 1) * i_L * (i_L + 1) / 2;

/*
    Consistent with the following for loop:
    int index = 0;
    for (int x = 0; x <= L; x++)
        for (int y = 0; x + y <= L; y++)
            for (int z = 0; x + y + z <= L; z++)
                index++;
*/
template <int L> requires (L >= 0)
static inline int triple_lower_triangular_index(const int x, const int y, const int z)
{
    return ((L + 1) * (L + 2) * (L + 3) - (L + 1 - x) * (L + 2 - x) * (L + 3 - x)) / 6 + ((L - x + 1) * (L - x + 2) - (L - x + 1 - y) * (L - x + 2 - y)) / 2 + z;
}

template <int L> requires (L >= 0)
constexpr int triple_lower_triangular_total = (L + 1) * (L + 2) * (L + 3) / 6;

/*
    Consistent with the following for loop:
    int index = 0;
    for (int x = L; x >= 0; x--)
        for (int y = L - x; y >= 0; y--)
        {
            const int z = L - x - y;
            index++;
        }
*/
template <int L> requires (L >= 0)
static inline int cartesian_orbital_index(const int x, const int y)
{
    return (L - x) * (L - x + 1) / 2 + L - x - y;
}

template <int L> requires (L >= 0)
constexpr int cartesian_orbital_total = lower_triangular_total<L>;
