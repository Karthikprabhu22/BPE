from BPE_functions import E, luminosity_distance
import numpy as np


def test_E():
    """
    Test functionality of fuction E(z, H_0, omega_m, omega_lam, omega_k)
    """
    test_1 = E(1.1, 1.2, 1.3, 1.4, 1.5)  # should be 4.4782027645027425
    test_2 = E(1.5, 1.4, 1.3, 1.2, 1.1)  # should be 5.327992117111285

    assert np.isclose(4.4782027645027425, test_1, rtol=0.0001)
    assert np.isclose(5.327992117111285, test_2, rtol=0.0001)


def test_luminosity_distance():
    """
    Test functionality of function 
    luminosity_distance(H_0, omega_m, omega_lam, omega_k, z)
    """
    H_0 = 74
    omega_m = 0.348
    omega_lam = 0.827
    z = 1
    test_1 = luminosity_distance(
        H_0, omega_m, omega_lam, 1.0 - omega_m - omega_lam, z)

    true_value = 6291.3  # Mpc from http://www.bo.astro.it/~cappi/cosmotools
    assert np.islcose(true_value, test_1, rtol=0.1)

