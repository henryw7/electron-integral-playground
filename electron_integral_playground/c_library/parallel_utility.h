
#pragma once

static inline void atomic_add(double* output, const double input)
{
    output[0] += input;
}
