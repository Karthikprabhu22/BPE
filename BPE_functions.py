# Import this list of functions to run MCMC scripts with the S18 data.
import random
import numpy as np
import os
from numpy.random import default_rng

rng = default_rng()
from scipy.integrate import quad


def invert_matrix(input_list):
    """
    Returns the inverse of input_list

    Parameters:
    input_list: 2d array
        an array of floats
    """
    inverted_matrix = np.linalg.inv(input_list)
    return inverted_matrix


def E(z, H_0, omega_m, omega_lam, omega_k):
    return 1/np.sqrt(omega_m * (1 + z) ** 3 + omega_k * (1 + z) ** 2 + omega_lam)


def hubble_distance(H_0):
    """
    Calculates the hubble distance for a given value of the hubble constant

    Parameters:
    H_0: float
        the hubble constant
    """
    speed_of_light = 299792.458  # speed of light in km/s
    return speed_of_light / H_0


def comoving_distance(H_0, omega_m, omega_lam, omega_k, z):
    """
    Calculates the comoving distance

    Parameters:
    H_0: float
        the Hubble constant
    omega_m: float
        total matter density
    omega_lam: float
        dark energy density
    omega_k: float
        curvature
    z: array
        redshift
    Returns:
    comoving_distance: float
        The comoving distance
    """

    comoving_distance = hubble_distance(H_0) * np.array(
        [quad(E, 0, i, args=(H_0, omega_m, omega_lam, omega_k))[0] for i in z]
    )
    return comoving_distance


def transverse_comoving_distance(H_0, omega_m, omega_lam, omega_k, z):
    """
    Calculates the transverse comoving distance

    Parameters:
    H_0: float
        the Hubble constant
    omega_m: float
        total matter density
    omega_lam: float
        dark energy density
    omega_k: float
        curvature
    z: float
        redshift
    """

    if omega_k > 0:
        return (
            hubble_distance(H_0)
            / np.sqrt(omega_k)
            * np.sinh(
                np.sqrt(omega_k)
                * comoving_distance(H_0, omega_m, omega_lam, omega_k, z)
                / hubble_distance(H_0)
            )
        )

    elif omega_k == 0:
        return comoving_distance(H_0, omega_m, omega_lam, omega_k, z)

    else:
        return (
            hubble_distance(H_0)
            / np.sqrt(np.abs(omega_k))
            * np.sin(
                np.sqrt(np.abs(omega_k))
                * comoving_distance(H_0, omega_m, omega_lam, omega_k, z)
                / hubble_distance(H_0)
            )
        )


def luminosity_distance(H_0, omega_m, omega_lam, omega_k, z):
    """
    Calculates the luminosity distance

    Parameters:
    H_0: float
        the Hubble constant
    omega_m: float
        total matter density
    omega_lam: float
        dark energy density
    omega_k: float
        curvature
    z: float
        redshift
    """
    return (1 + z) * transverse_comoving_distance(H_0, omega_m, omega_lam, omega_k, z)


def signal(H_0, omega_m, omega_lam, omega_k, z):
    """
    Calculates the signal

    Parameters:
    H_0: float
        the Hubble constant
    omega_m: float
        total matter density
    omega_lam: float
        dark energy density
    omega_k: float
        curvature
    z: float
        redshift
    """
    return 5 * np.log10(luminosity_distance(H_0, omega_m, omega_lam, omega_k, z)*10**6 / 10)


def chi_squared(H_0, omega_m, omega_lam, omega_k, M, container):
    """
    Calculates chi squared

    Parameters:
    H_0: float
        the Hubble constant
    omega_m: float
        total matter density
    omega_lam: float
        dark energy density
    omega_k: float
        curvature
    M: float
        Nuisance parameter
    z: array
        redshift
    container: DataContainer
        A container filled with data imported from the provided data files.

    Returns:
    chi_squared: float
        The chi_squared value for that particular point in parameter space.
    """

    z = container.z
    mb = container.mb
    covariance_matrix = container.covariance_matrix
    inverted_covariance_matrix = invert_matrix(covariance_matrix)

    chi_squared = np.linalg.multi_dot(
        [
            (mb - (signal(H_0, omega_m, omega_lam, omega_k, z) + M)),
            inverted_covariance_matrix,
            (mb - (signal(H_0, omega_m, omega_lam, omega_k, z) + M)),
        ]
    )
    return chi_squared


def likelihood(H_0, omega_m, omega_lam, omega_k, M, container):
    return np.exp(-chi_squared(H_0, omega_m, omega_lam, omega_k, M, container) / 2)


def generating_function(param_vector, mcmc_covariance=np.diag([2.5, 0.03, 0.03, 0.5])):

    mean = param_vector
    cov = mcmc_covariance

    new_state = np.random.multivariate_normal(mean, cov)
    return new_state


def metropolis(current_state, container):
    """
    Perform one step of the metropolis algorithm, does not move time forward.
    The generating function is tbd.
    current_state[0]=H_0
    current_state[1]=Omega_m
    current_state[2]=Omega_lam
    current_state[3]=M
    """
    r = np.random.random()
    g_vector = generating_function(current_state)
    ratio = likelihood(
        g_vector[0],
        g_vector[1],
        g_vector[2],
        1.0 - g_vector[1] - g_vector[2],
        g_vector[3],
        container,
    ) / (
        likelihood(
            current_state[0],
            current_state[1],
            current_state[2],
            1.0 - current_state[1] - current_state[2],
            current_state[3],
            container,
        )
    )
    if ratio >= 1:
        return g_vector
    if ratio < r:
        return current_state
    if ratio > r:
        return g_vector


def MCMC(num_iter, container):
    """
    Run the Markov Chain Monte Carlo algorithm for num_iter steps on the likelihood distribution.
    """
    # create the random initial configuration in parameter space
    current_state = [
        np.random.normal(loc=70, scale=3),
        np.random.normal(loc=0.3, scale=0.0001),
        np.random.normal(loc=0.7, scale=0.0001),
        np.random.normal(loc=-18, scale=0.01),
    ]
    chain = [current_state]
    for _ in range(num_iter):
        link = metropolis(current_state, container)
        chain.append(link)
        current_state = link
    # Don't include the beginning of the chain to ensure a steady state.
    return chain


class DataContainer(object):
    """
    A class used to import and store data of the form expected for this project
    """

    def __init__(self):
        self.name = []
        self.z = []
        self.mb = []
        self.dmb = []
        self.systematic_covariance_matrix = []
        self.covariance_matrix = []

    def import_params(self):
        """
        Imports the parameter values from the file lcparam_DS17f.txt and stores them in separate arrays, ordered by name (in this case name is an integer 0-39)
        lcparam_DS17f.txt is assumed to be stored in a directory called data that is located in the same directory as BPE_function.py
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath = dir_path + "/data/lcparam_DS17f.txt"

        self.name, self.z, self.mb, self.dmb = np.genfromtxt(
            filepath, usecols=(0, 1, 4, 5), delimiter=" ", unpack=True
        )

    def import_systematic_covariance_matrix(self):
        """
        Imports the systematic covariance matrix from the file sys_DS17f.txt and stores it in a 2D array
        """

        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath = dir_path + "/data/sys_DS17f.txt"

        covariance_values = []

        with open(filepath) as file:
            matrix_dimension = int(file.readline())
            for line in file:
                value = float(line)
                covariance_values.append(value)

        covariance_list = np.asarray(covariance_values)
        two_d_covariance_matrix = covariance_list.reshape(
            matrix_dimension, matrix_dimension
        )

        self.systematic_covariance_matrix = two_d_covariance_matrix

    def calculate_total_covariance_matrix(self):
        self.covariance_matrix = np.copy(self.systematic_covariance_matrix)
        for i in range(40):
            self.covariance_matrix[i][i] += self.dmb[i] ** 2


def Ez(z, omega_m, omega_lam, omega_k):
    return np.sqrt(omega_m * (1 + z) ** 3 + omega_lam + omega_k * (1 + z) ** 2)


def integrate_Ez_prime(z, num_bins, omega_m, omega_lam, omega_k):
    step_size = z / num_bins
    Total = 0
    z_prime = 0

    for _ in range(len(num_bins)):
        Total += (
            (
                Ez(z_prime + step_size, omega_m, omega_lam, omega_k)
                + Ez(z_prime + step_size, omega_m, omega_lam, omega_k)
            )
            * step_size
            / 2
        )
        z_prime += step_size
    return Total


def mu_data(m_B, M):
    r"""
    Function to return $\mu^d$, the value of distance modulus inferred from the data.
    Based on equation (3) in the paper.
    Parameters
    ----------
    m_B: float
        Log of the overall flux normalization. Available in the dataset.
    
    M: float
        The absolute B-band magnitude of a fiducial SN Ia with x1 = 0 and c = 0. 
        Nuisance parameter that needs to be sampled.

    Returns
    -------
    mu: float
        The distance modulus inferred from the data.
    """
    mu = m_B - M
    return mu
