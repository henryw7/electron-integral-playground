
import pytest

import numpy as np

from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis

basis_read_threshold = 0

def test_read_gaussian_basis_file_321g():
    basis_name = "3-21g"
    basis_set = read_basis(basis_name)
    li_shells = basis_set["Li"]

    assert len(li_shells) == 5
    assert li_shells[0].angular == 0
    np.testing.assert_allclose(li_shells[0].primitive_exponents, [0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01], atol = basis_read_threshold)
    np.testing.assert_allclose(li_shells[0].primitive_coefficients, [0.6966866381E-01, 0.3813463493E+00, 0.6817026244E+00], atol = basis_read_threshold)
    assert li_shells[1].angular == 0
    np.testing.assert_allclose(li_shells[1].primitive_exponents, [0.5402050000E+00, 0.1022550000E+00], atol = basis_read_threshold)
    np.testing.assert_allclose(li_shells[1].primitive_coefficients, [-0.2631264058E+00, 0.1143387418E+01], atol = basis_read_threshold)
    assert li_shells[2].angular == 1
    np.testing.assert_allclose(li_shells[2].primitive_exponents, [0.5402050000E+00, 0.1022550000E+00], atol = basis_read_threshold)
    np.testing.assert_allclose(li_shells[2].primitive_coefficients, [0.1615459708E+00, 0.9156628347E+00], atol = basis_read_threshold)
    assert li_shells[3].angular == 0
    np.testing.assert_allclose(li_shells[3].primitive_exponents, [0.2856450000E-01], atol = basis_read_threshold)
    np.testing.assert_allclose(li_shells[3].primitive_coefficients, [1.0], atol = basis_read_threshold)
    assert li_shells[4].angular == 1
    np.testing.assert_allclose(li_shells[4].primitive_exponents, [0.2856450000E-01], atol = basis_read_threshold)
    np.testing.assert_allclose(li_shells[4].primitive_coefficients, [1.0], atol = basis_read_threshold)

def test_read_gaussian_basis_file_def2tzvp():
    basis_name = "def2-TZVP"
    basis_set = read_basis(basis_name)
    c_shells = basis_set["C"]

    assert len(c_shells) == 11
    assert c_shells[0].angular == 0
    np.testing.assert_allclose(c_shells[0].primitive_exponents, [13575.3496820, 2035.2333680, 463.22562359, 131.20019598, 42.853015891, 15.584185766], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[0].primitive_coefficients, [0.22245814352E-03, 0.17232738252E-02, 0.89255715314E-02, 0.35727984502E-01, 0.11076259931, 0.24295627626], atol = basis_read_threshold)
    assert c_shells[1].angular == 0
    np.testing.assert_allclose(c_shells[1].primitive_exponents, [6.2067138508, 2.5764896527], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[1].primitive_coefficients, [0.41440263448, 0.23744968655], atol = basis_read_threshold)
    assert c_shells[2].angular == 0
    np.testing.assert_allclose(c_shells[2].primitive_exponents, [0.57696339419], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[2].primitive_coefficients, [1.0], atol = basis_read_threshold)
    assert c_shells[3].angular == 0
    np.testing.assert_allclose(c_shells[3].primitive_exponents, [0.22972831358], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[3].primitive_coefficients, [1.0], atol = basis_read_threshold)
    assert c_shells[4].angular == 0
    np.testing.assert_allclose(c_shells[4].primitive_exponents, [0.95164440028E-01], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[4].primitive_coefficients, [1.0], atol = basis_read_threshold)
    assert c_shells[5].angular == 1
    np.testing.assert_allclose(c_shells[5].primitive_exponents, [34.697232244, 7.9582622826, 2.3780826883, 0.81433208183], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[5].primitive_coefficients, [0.53333657805E-02, 0.35864109092E-01, 0.14215873329, 0.34270471845], atol = basis_read_threshold)
    assert c_shells[6].angular == 1
    np.testing.assert_allclose(c_shells[6].primitive_exponents, [0.28887547253], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[6].primitive_coefficients, [0.46445822433], atol = basis_read_threshold)
    assert c_shells[7].angular == 1
    np.testing.assert_allclose(c_shells[7].primitive_exponents, [0.10056823671], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[7].primitive_coefficients, [0.24955789874], atol = basis_read_threshold)
    assert c_shells[8].angular == 2
    np.testing.assert_allclose(c_shells[8].primitive_exponents, [1.09700000], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[8].primitive_coefficients, [1.0], atol = basis_read_threshold)
    assert c_shells[9].angular == 2
    np.testing.assert_allclose(c_shells[9].primitive_exponents, [0.31800000], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[9].primitive_coefficients, [1.0], atol = basis_read_threshold)
    assert c_shells[10].angular == 3
    np.testing.assert_allclose(c_shells[10].primitive_exponents, [0.76100000], atol = basis_read_threshold)
    np.testing.assert_allclose(c_shells[10].primitive_coefficients, [1.0], atol = basis_read_threshold)
