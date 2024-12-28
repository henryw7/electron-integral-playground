
import sys
from math import sqrt

assert len(sys.argv) == 2

n = float(sys.argv[1])

n2 = n**2
found = False
closest_diff = 1.0
for denominator in range(1, 10000):
    numerator = n2 * denominator
    rounded_numerator = round(numerator)
    diff = abs(numerator - rounded_numerator)
    if diff < 1e-10:
        print(f"{n} = sqrt({rounded_numerator} / {denominator}), error = {diff:2e}")
        found = True
        break
    closest_diff = min(diff, closest_diff)

if not found:
    print(f"not found, closest_diff = {closest_diff}")
