
#pragma once

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
