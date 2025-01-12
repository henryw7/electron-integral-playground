
import pytest

from electron_integral_code_generator.equation_mcmurchie_davidson import get_mcmurchie_davidson_E_term, get_mcmurchie_davidson_R_term
from electron_integral_code_generator.sympy_utility import remove_whitespace

def test_mcmurchie_davidson_E_terms():
    E_0_0_0_x = get_mcmurchie_davidson_E_term(0, 0, 0, "x")
    assert E_0_0_0_x == "1"
    E_0_0_0_y = get_mcmurchie_davidson_E_term(0, 0, 0, "y")
    assert E_0_0_0_y == "1"
    E_1_0_0_x = get_mcmurchie_davidson_E_term(1, 0, 0, "x")
    assert E_1_0_0_x == "PAx"
    E_1_0_0_y = get_mcmurchie_davidson_E_term(1, 0, 0, "y")
    assert E_1_0_0_y == "PAy"
    E_1_0_1_y = get_mcmurchie_davidson_E_term(1, 0, 1, "y")
    assert E_1_0_1_y == "one_over_two_p"
    E_0_1_0_y = get_mcmurchie_davidson_E_term(0, 1, 0, "y")
    assert E_0_1_0_y == "PBy"
    E_0_1_1_y = get_mcmurchie_davidson_E_term(0, 1, 1, "y")
    assert E_0_1_1_y == "one_over_two_p"
    E_1_1_0_y = get_mcmurchie_davidson_E_term(1, 1, 0, "y")
    assert remove_whitespace(E_1_1_0_y) == remove_whitespace("PAy * PBy + one_over_two_p")
    E_1_1_1_y = get_mcmurchie_davidson_E_term(1, 1, 1, "y")
    assert remove_whitespace(E_1_1_1_y) == remove_whitespace("(PAy + PBy) * one_over_two_p") or \
           remove_whitespace(E_1_1_1_y) == remove_whitespace("one_over_two_p * (PAy + PBy)")
    E_1_1_2_y = get_mcmurchie_davidson_E_term(1, 1, 2, "y")
    assert remove_whitespace(E_1_1_2_y) == remove_whitespace("one_over_two_p * one_over_two_p")

    E_3_3_0_z = get_mcmurchie_davidson_E_term(3, 3, 0, "z")
    assert remove_whitespace(E_3_3_0_z) == remove_whitespace("PAz*PAz*PAz*PBz*PBz*PBz + 3*PAz*PAz*PAz*PBz*one_over_two_p + 9*PAz*PAz*PBz*PBz*one_over_two_p + 9*PAz*PAz*one_over_two_p*one_over_two_p + 3*PAz*PBz*PBz*PBz*one_over_two_p + 27*PAz*PBz*one_over_two_p*one_over_two_p + 9*PBz*PBz*one_over_two_p*one_over_two_p + 15*one_over_two_p*one_over_two_p*one_over_two_p")
    E_3_3_3_z = get_mcmurchie_davidson_E_term(3, 3, 3, "z")
    assert remove_whitespace(E_3_3_3_z) == remove_whitespace("one_over_two_p*one_over_two_p*one_over_two_p*(PAz*PAz*PAz + 9*PAz*PAz*PBz + 9*PAz*PBz*PBz + 30*PAz*one_over_two_p + PBz*PBz*PBz + 30*PBz*one_over_two_p)")
    E_3_3_6_z = get_mcmurchie_davidson_E_term(3, 3, 6, "z")
    assert remove_whitespace(E_3_3_6_z) == remove_whitespace("one_over_two_p*one_over_two_p*one_over_two_p*one_over_two_p*one_over_two_p*one_over_two_p")

def test_mcmurchie_davidson_R_terms():
    R_0_0_0_0 = get_mcmurchie_davidson_R_term(0, 0, 0)
    assert R_0_0_0_0 == "R_0_0_0_0"
    R_0_0_0_10 = get_mcmurchie_davidson_R_term(0, 0, 0, 10)
    assert R_0_0_0_10 == "R_0_0_0_10"
    R_1_0_0_0 = get_mcmurchie_davidson_R_term(1, 0, 0)
    assert remove_whitespace(R_1_0_0_0) == remove_whitespace("PQx * R_0_0_0_1")
    R_0_1_0_0 = get_mcmurchie_davidson_R_term(0, 1, 0)
    assert remove_whitespace(R_0_1_0_0) == remove_whitespace("PQy * R_0_0_0_1")
    R_0_0_1_0 = get_mcmurchie_davidson_R_term(0, 0, 1)
    assert remove_whitespace(R_0_0_1_0) == remove_whitespace("PQz * R_0_0_0_1")

    R_1_1_0_0 = get_mcmurchie_davidson_R_term(1, 1, 0)
    assert remove_whitespace(R_1_1_0_0) == remove_whitespace("PQx * PQy * R_0_0_0_2")
    R_2_0_0_0 = get_mcmurchie_davidson_R_term(2, 0, 0)
    assert remove_whitespace(R_2_0_0_0) == remove_whitespace("PQx * PQx * R_0_0_0_2 + R_0_0_0_1")
    R_3_0_0_0 = get_mcmurchie_davidson_R_term(3, 0, 0)
    assert remove_whitespace(R_3_0_0_0) == remove_whitespace("PQx*(PQx*PQx*R_0_0_0_3 + 3*R_0_0_0_2)")
    R_4_0_0_0 = get_mcmurchie_davidson_R_term(4, 0, 0)
    print(R_4_0_0_0)
    assert remove_whitespace(R_4_0_0_0) == remove_whitespace("PQx*PQx*PQx*PQx*R_0_0_0_4 + 6*PQx*PQx*R_0_0_0_3 + 3*R_0_0_0_2")

