
#pragma once

static inline void atomic_add(double* output, const double input)
{
#pragma omp atomic update
    output[0] += input;
}
