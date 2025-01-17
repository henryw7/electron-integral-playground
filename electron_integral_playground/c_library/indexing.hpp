
#pragma once

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
    Consistent with the following for loop:
    int index;
    for (int row = 0; row < M; row++)
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
    int index;
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
    int index;
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
