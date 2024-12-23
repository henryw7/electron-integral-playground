
import numpy as np

from electron_integral_playground.data_structure import Molecule, PrimitivePair, PrimitivePairList, PrimitivePairData, PrimitivePairDataAngularList

def form_primitive_pair_list(molecule: Molecule, schwarz_upper_bound: float) -> PrimitivePairList:
    """
    Record all primitive shell pairs whose upper bound >= schwarz_upper_bound
    The computed upper bound is the maximum of Schwarz upper bound and overlap integral upper bound,
    so the resulting list of pairs is sufficient for both overlap and Coulomb integrals.
    """
    r"""
    For an arbitrary Cartesian Gaussian function along one dimension
    $$\mu(x) = (x - A_x)^{i_x} e^{-a (x - A_x)^2}$$
    At region $x > A_x$, the following inequality holds:
    $$(x - A_x)^{i_x} e^{-a (x - A_x)^2} \leq \left(\frac{i_x}{2ae}\right)^{i_x / 2} e^{-a \left(x - A_x - \sqrt{\frac{i_x}{2a}}\right)^2}$$
    Here we define $i_x^{i_x} = 1$ if $i_x = 0$.

    The proof is: Since both side of the inequality is greater than zero in the region $x > A_x$, we can define the ratio function
    \begin{align*}
        R(x) &= \frac{ (x - A_x)^{i_x} e^{-a (x - A_x)^2} }{ \left(\frac{i_x}{2ae}\right)^{i_x / 2} e^{-a \left(x - A_x - \sqrt{\frac{i_x}{2a}}\right)^2} } \\
            &= \frac{ (x - A_x)^{i_x} }{ \left(\frac{i_x}{2ae}\right)^{i_x / 2} e^{2a(x - A_x)\sqrt{\frac{i_x}{2a}}} e^{- \frac{i_x}{2}} } \\
            &= \frac{ (x - A_x)^{i_x} }{ \sqrt{\frac{i_x}{2a}}^{i_x} e^{\sqrt{2ai_x} (x - A_x)} e^{-i_x} } \\
            &= \frac{ e^{i_x ln(x - A_x)} }{ e^{i_x ln\left(\sqrt{\frac{i_x}{2a}}\right)} e^{\sqrt{2ai_x} (x - A_x)} e^{-i_x} } \\
            &= e^{i_x ln(x - A_x) - i_x ln\left(\sqrt{\frac{i_x}{2a}}\right) - \sqrt{2ai_x} (x - A_x) + i_x} \\
            &= e^{i_x ln(\sqrt{2ai_x} (x - A_x)) - \sqrt{2ai_x} (x - A_x) + i_x - i_x ln(i_x)}
    \end{align*}
    The exponent of the ratio function is
    $$ln(R(x)) = i_x ln(\sqrt{2ai_x} (x - A_x)) - \sqrt{2ai_x} (x - A_x) + i_x - i_x ln(i_x)$$
    It maximize at $x_{max} = A_x + \sqrt{\frac{i_x}{2a}}$, where $\left. \frac{d}{dx} ln(R(x)) \right|_{x = x_{max}} = 0$ and $ln(R(x_{max})) = 0$, and for any other points in $x > A_x$, $ln(R(x)) < ln(R(x_{max})) = 0$. So $R(x) \leq 1$ for $x > A_x$, and thus the inequality holds.

    For the pair product of two primitive Gaussian functions along one direction
    $$(x-A_x)^{i_x} (x-B_x)^{j_x} e^{-a (x - A_x)^2} e^{-b (x - B_x)^2}$$
    Assume $A_x \leq B_x$. If $|A_x - B_x| \geq \sqrt{\frac{i_x}{2a}} + \sqrt{\frac{j_x}{2b}}$, then we can bound the product by
    \begin{align*}
        &\quad (x-A_x)^{i_x} (x-B_x)^{j_x} e^{-a (x - A_x)^2} e^{-b (x - B_x)^2} \\
        &\leq \left(\frac{i_x}{2ae}\right)^{i_x / 2} e^{-a \left(x - A_x - \sqrt{\frac{i_x}{2a}}\right)^2} \left(\frac{j_x}{2be}\right)^{j_x / 2} e^{-b \left(x - B_x + \sqrt{\frac{j_x}{2b}}\right)^2} \\
        &= \left(\frac{i_x}{2ae}\right)^{i_x / 2} \left(\frac{j_x}{2be}\right)^{j_x / 2} e^{-\frac{ab}{a+b} \left(|A_x - B_x| - \sqrt{\frac{i_x}{2a}} - \sqrt{\frac{j_x}{2b}}\right)^2} e^{-(a+b) (x - P_x)^2}
    \end{align*}
    where $\vec{P}$ is a point between $\vec{A}$ and $\vec{B}$. \par
    Otherwise we assume the maximum overlap between the two Gaussian functions ($\vec{A} = \vec{B} = \vec{P}$), and the product is bounded by
    $$(x-A_x)^{i_x} (x-B_x)^{j_x} e^{-a (x - A_x)^2} e^{-b (x - B_x)^2} \leq 2\left(\frac{i_x}{2ae}\right)^{i_x / 2} \left(\frac{j_x}{2be}\right)^{j_x / 2} e^{-(a+b) (x - P_x)^2}$$
    The factor of 2 comes from the overlap in both sides ($x < A_x$ and $x > A_x$).

    For the pair product of any Gaussian function combination of two primitive shells
    $$\mu\nu = (x-A_x)^{i_x} (y-A_y)^{i_y} (z-A_z)^{i_z} (x-B_x)^{j_x} (y-B_y)^{j_y} (z-B_z)^{j_z} e^{-a \left|\vec{r} - \vec{A}\right|^2} e^{-b \left|\vec{r} - \vec{B}\right|^2}$$
    where $i_x + i_y + i_z = L_\mu$ and $j_x + j_y + j_z = L_\nu$. Since
    $$\left(\frac{i_x}{2ae}\right)^{i_x / 2} \left(\frac{i_y}{2ae}\right)^{i_y / 2} \left(\frac{i_z}{2ae}\right)^{i_z / 2} \leq \left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2}$$
    If $|\vec{A} - \vec{B}| \geq \sqrt{\frac{L_\mu}{2a}} + \sqrt{\frac{L_\nu}{2b}}$, the pair product is bounded by
    $$\mu\nu \leq \left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-\frac{ab}{a+b} \left(|\vec{A} - \vec{B}| - \sqrt{\frac{L\mu}{2a}} - \sqrt{\frac{L_\nu}{2b}}\right)^2} e^{-(a+b) |\vec{r} - \vec{P}|^2}$$
    else the pair product is bounded by the maximum overlap case
    $$\mu\nu \leq 2\left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-(a+b) |\vec{r} - \vec{P}|^2}$$
    The $\vec{P}$ in both inequalities refer to some point on the line of $\vec{A}$ and $\vec{B}$. \par
    To make the upper bound function smooth, we also apply the factor of 2 if $|\vec{A} - \vec{B}| \geq \sqrt{\frac{L_\mu}{2a}} + \sqrt{\frac{L_\nu}{2b}}$, so
    $$\mu\nu \leq 2\left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-\frac{ab}{a+b} \left(\max\left( |\vec{A} - \vec{B}| - \sqrt{\frac{L\mu}{2a}} - \sqrt{\frac{L_\nu}{2b}}, 0 \right)\right)^2} e^{-(a+b) |\vec{r} - \vec{P}|^2}$$

    The overlap upper bound of a pair of primitive shell is
    \begin{align*}
        \iiint_\infty d\vec{r} \mu(\vec{r})\nu(\vec{r}) &\leq 2\left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-\frac{ab}{a+b} \left(\max\left( |\vec{A} - \vec{B}| - \sqrt{\frac{L\mu}{2a}} - \sqrt{\frac{L_\nu}{2b}}, 0 \right)\right)^2} \iiint_\infty d\vec{r} e^{-(a+b) |\vec{r} - \vec{P}|^2} \\
            &= 2\left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-\frac{ab}{a+b} \left(\max\left( |\vec{A} - \vec{B}| - \sqrt{\frac{L\mu}{2a}} - \sqrt{\frac{L_\nu}{2b}}, 0 \right)\right)^2} \left(\frac{\pi}{a+b}\right)^{3/2}
    \end{align*}

    The Schwarz upper bound of a pair of primitive shell involves evaluating integral $\sqrt{(\mu\nu|\mu\nu)}$, which is bounded by
    \begin{align*}
        (\mu\nu|\mu\nu) &= \iiint_\infty d\vec{r}_1 \iiint_\infty d\vec{r}_2 \mu(\vec{r}_1)\nu(\vec{r}_1) \frac{1}{|\vec{r}_1 - \vec{r}_2|} \mu(\vec{r}_2)\nu(\vec{r}_2) \\
            &\leq \left( 2\left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-\frac{ab}{a+b} \left(\max\left( |\vec{A} - \vec{B}| - \sqrt{\frac{L\mu}{2a}} - \sqrt{\frac{L_\nu}{2b}}, 0 \right)\right)^2} \right)^2 \\
                &\quad \iiint_\infty d\vec{r} e^{-(a+b) |\vec{r}_1 - \vec{P}|^2} e^{-(a+b) |\vec{r}_2 - \vec{P}|^2} \frac{1}{|\vec{r}_1 - \vec{r}_2|} \\
            &= \left( 2\left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-\frac{ab}{a+b} \left(\max\left( |\vec{A} - \vec{B}| - \sqrt{\frac{L\mu}{2a}} - \sqrt{\frac{L_\nu}{2b}}, 0 \right)\right)^2} \right)^2 \frac{2 \pi^{5/2}}{(a+b)^2 \sqrt{2(a+b)}}
    \end{align*}
    The last equation follows from the formula of $(ss|ss)$ integral, and noticing that $F_0(0) = 1$. So
    $$\sqrt{(\mu\nu|\mu\nu)} \leq 2\left(\frac{L_\mu}{2ae}\right)^{L_\mu / 2} \left(\frac{L_\nu}{2be}\right)^{L_\nu / 2} e^{-\frac{ab}{a+b} \left(\max\left( |\vec{A} - \vec{B}| - \sqrt{\frac{L\mu}{2a}} - \sqrt{\frac{L_\nu}{2b}}, 0 \right)\right)^2} \frac{2^{1/4} \pi^{5/4}}{(a+b)^{5/4}}$$
    """

    assert molecule.basis_shells is not None and len(molecule.basis_shells) > 0
    assert molecule.n_ao > 0

    pair_list = {}
    for i_shell, shell_i in enumerate(molecule.basis_shells):
        i_angular = shell_i.angular
        position_A = molecule.geometry[shell_i.i_atom, :]
        for j_shell, shell_j in enumerate(molecule.basis_shells):
            j_angular = shell_j.angular
            position_B = molecule.geometry[shell_j.i_atom, :]

            distance_AB = np.linalg.norm(position_A - position_B)

            for i_primitive in range(len(shell_i.primitive_exponents)):
                exponent_a = shell_i.primitive_exponents[i_primitive]
                coefficient_A = np.abs(shell_i.primitive_coefficients[i_primitive])

                if i_angular > 0:
                    coefficient_A *= np.power(i_angular / (2.0 * np.e * exponent_a), i_angular / 2.0)
                distance_AB -= np.sqrt(i_angular / 2.0 / exponent_a)

                for j_primitive in range(len(shell_j.primitive_exponents)):
                    exponent_b = shell_j.primitive_exponents[j_primitive]
                    coefficient_B = np.abs(shell_j.primitive_coefficients[j_primitive])

                    if j_angular > 0:
                        coefficient_B *= np.power(j_angular / (2.0 * np.e * exponent_b), j_angular / 2.0)
                    distance_AB -= np.sqrt(j_angular / 2.0 / exponent_b)

                    distance_AB = max(distance_AB, 0.0)
                    gaussian_AB = np.exp(-exponent_a * exponent_b / (exponent_a + exponent_b) * distance_AB)
                    coefficient_AB = 2.0 * coefficient_A * coefficient_B
                    overlap_factor = np.power(np.pi / (exponent_a + exponent_b), 1.5)
                    schwarz_factor = np.power(np.pi / (exponent_a + exponent_b), 1.25) * np.power(2, 0.25)
                    total_upper_bound = coefficient_AB * gaussian_AB * max(overlap_factor, schwarz_factor)

                    if total_upper_bound < schwarz_upper_bound:
                        continue

                    ij_angular = (i_angular, j_angular)
                    if ij_angular not in pair_list:
                        pair_list[ij_angular] = []
                    pair_list[ij_angular].append(PrimitivePair(
                        i_shell = i_shell,
                        j_shell = j_shell,
                        i_primitive = i_primitive,
                        j_primitive = j_primitive,
                        upper_bound = total_upper_bound,
                    ))

    for ij_angular in pair_list:
        pair_list[ij_angular].sort(key = lambda pair: pair.upper_bound, reverse = True) # Larger upper_bound comes first

    return pair_list

def spherical_bool_pair_to_one_int(i_spherical: bool, j_spherical: bool) -> int:
    return (2 if i_spherical else 0) + (1 if j_spherical else 0)

def form_primitive_pair_data(molecule: Molecule, pair_list: PrimitivePairList) -> PrimitivePairDataAngularList:
    assert molecule.basis_shells is not None and len(molecule.basis_shells) > 0
    assert molecule.n_ao > 0

    pair_data_list = {}
    for ij_angular in pair_list:
        n_pair = len(pair_list[ij_angular])
        pair_data = PrimitivePairData(
            P_p = np.empty((n_pair, 4), dtype = np.float64),
            A_a = np.empty((n_pair, 4), dtype = np.float64),
            B_b = np.empty((n_pair, 4), dtype = np.float64),
            coefficient = np.empty(n_pair, dtype = np.float64),
            i_ao_start = np.empty(n_pair, dtype = np.int32),
            j_ao_start = np.empty(n_pair, dtype = np.int32),
            i_atom = np.empty(n_pair, dtype = np.int32),
            j_atom = np.empty(n_pair, dtype = np.int32),
            ij_spherical = np.empty(n_pair, dtype = np.int32),
        )
        for i_pair in range(n_pair):
            pair = pair_list[ij_angular][i_pair]
            shell_i = molecule.basis_shells[pair.i_shell]
            shell_j = molecule.basis_shells[pair.j_shell]
            position_A = molecule.geometry[shell_i.i_atom, :]
            position_B = molecule.geometry[shell_j.i_atom, :]
            exponent_a = shell_i.primitive_exponents[pair.i_primitive]
            exponent_b = shell_j.primitive_exponents[pair.j_primitive]
            coefficient_A = shell_i.primitive_coefficients[pair.i_primitive]
            coefficient_B = shell_j.primitive_coefficients[pair.j_primitive]

            exponent_p = exponent_a + exponent_b
            position_P = (exponent_a * position_A + exponent_b * position_B) / exponent_p
            distance_AB = np.linalg.norm(position_A - position_B)
            gaussian_AB = np.exp(-exponent_a * exponent_b / (exponent_a + exponent_b) * distance_AB)
            total_coefficient = coefficient_A * coefficient_B * gaussian_AB

            pair_data.A_a[i_pair, 0:3] = position_A
            pair_data.A_a[i_pair, 3]   = exponent_a
            pair_data.B_b[i_pair, 0:3] = position_B
            pair_data.B_b[i_pair, 3]   = exponent_b
            pair_data.P_p[i_pair, 0:3] = position_P
            pair_data.P_p[i_pair, 3]   = exponent_p
            pair_data.coefficient[i_pair] = total_coefficient
            pair_data.i_ao_start[i_pair] = shell_i.i_ao_start
            pair_data.j_ao_start[i_pair] = shell_j.i_ao_start
            pair_data.i_atom[i_pair] = shell_i.i_atom
            pair_data.j_atom[i_pair] = shell_j.i_atom
            pair_data.ij_spherical[i_pair] = spherical_bool_pair_to_one_int(shell_i.spherical, shell_j.spherical)

        pair_data_list[ij_angular] = pair_data

    return pair_data_list
