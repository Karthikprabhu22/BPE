from BPE_functions import (
    E,
    luminosity_distance,
    comoving_distance,
    signal,
    DataContainer,
    generating_function,
    metropolis,
)
import numpy as np


def test_E():
    """
    Test functionality of function E(z, H_0, omega_m, omega_lam, omega_k)
    """
    test_1 = E(1.1, 1.2, 1.3, 1.4, 1.5)  # should be 0.22330386822291187
    test_2 = E(1.5, 1.4, 1.3, 1.2, 1.1)  # should be 0.18768796537600305

    assert np.isclose(
        0.22330386822291187, test_1, rtol=0.0001
    ), "something wrong with the E function"
    assert np.isclose(
        0.18768796537600305, test_2, rtol=0.0001
    ), "Something wrong with the E function"


def test_luminosity_distance():
    """
    Test functionality of function 
    luminosity_distance(H_0, omega_m, omega_lam, omega_k, z)
    """
    H_00 = 74
    omega_m0 = 0.348
    omega_lam0 = 0.827
    z0 = np.array([1.0])
    test_1 = luminosity_distance(
        H_00, omega_m0, omega_lam0, 1.0 - omega_m0 - omega_lam0, z0
    )

    true_value = 6291.3  # Mpc from http://www.bo.astro.it/~cappi/cosmotools
    assert np.isclose(
        true_value, test_1, rtol=0.1
    ), "something wrong with the luminosity_distance function"


def test_generating_function():
    container = DataContainer()
    param_vector = [0.1, 0.2, 0.3, 0.4]
    test_g_f = generating_function(
        param_vector, container, include_systematic_errors=True, include_M_prior=False
    )
    assert len(test_g_f) == 4
    for parameter in test_g_f:
        assert type(parameter) == float


def test_metropolis():
    container = DataContainer()
    current_state = [0.1, 0.2, 0.3, 0.4]
    new_state = metropolis(
        current_state, container, include_systematic_errors=True, include_M_prior=False
    )
    assert len(metropolis) == 4
    for parameter in new_state:
        assert type(parameter) == float


test_E()
test_luminosity_distance()
test_generating_function
