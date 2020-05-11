# Import this list of functions to run MCMC scripts with the S18 data.
import random
import numpy as np
import os

from scipy.integrate import quad


def E(z, H_0, omega_m, omega_lam, omega_k):
    # returns 1/E(z)

    E_val = 1 / np.sqrt(omega_m * (1 + z) ** 3 + omega_k * (1 + z) ** 2 + omega_lam)
    return E_val


def hubble_distance(H_0):
    """
    Calculates the hubble distance for a given value of the hubble constant

    Parameters:
    H_0: float
        the hubble constant in km/(s*Mpc)
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
    signal = 5 * np.log10(
        luminosity_distance(H_0, omega_m, omega_lam, omega_k, z) * 10 ** 6 / 10
    )

    return signal


def chi_squared(
    H_0, omega_m, omega_lam, omega_k, M, container, include_systematic_errors=True
):
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
    if include_systematic_errors:
        inverted_covariance_matrix = container.inverted_covariance_matrix
    else:
        inverted_covariance_matrix = container.inverted_statistical_covariance_matrix

    chi_squared = np.linalg.multi_dot(
        [
            (mb - (signal(H_0, omega_m, omega_lam, omega_k, z) + M)),
            inverted_covariance_matrix,
            (mb - (signal(H_0, omega_m, omega_lam, omega_k, z) + M)),
        ]
    )
    return chi_squared


def likelihood(
    H_0, omega_m, omega_lam, omega_k, M, container, include_systematic_errors=True
):
    """
    returns the likelihood of a state given a set of parameters.
    """

    return np.exp(
        -chi_squared(
            H_0, omega_m, omega_lam, omega_k, M, container, include_systematic_errors
        )
        / 2
    )


def generating_function(
    param_vector,
    container,
    mcmc_covariance=np.diag([0.1, 0.001, 0.001, 0.01]),
    include_systematic_errors=True,
):
    """
    creates a new state by sampling from a multivariate normal distribution around the current state with covariance matrix possibly given by the user.
    Does not allow the new state to be outside the realm of physical possibility.
    """

    mean = param_vector
    cov = mcmc_covariance

    new_state = np.random.multivariate_normal(mean, cov)
    if new_state[1] < 0 or new_state[2] < 0:
        new_state = param_vector

    return new_state


def metropolis(
    current_state, container, include_systematic_errors=True, include_M_prior=False
):
    """
    Perform one step of the metropolis algorithm, does not move time forward.
    The generating function is tbd.
    current_state[0]=H_0
    current_state[1]=Omega_m
    current_state[2]=Omega_lam
    current_state[3]=M
    """
    r = np.random.random()

    g_vector = generating_function(
        current_state, container, include_systematic_errors=include_systematic_errors
    )

    if include_M_prior:
        prior_g = np.exp(-((-19.23 - g_vector[3]) ** 2) / (2 * 0.042 ** 2))
        prior_current = np.exp(-((-19.23 - current_state[3]) ** 2) / (2 * 0.042 ** 2))
        prior_ratio = prior_g / prior_current
    else:
        prior_ratio = 1

    ratio = (
        prior_ratio
        * likelihood(
            g_vector[0],
            g_vector[1],
            g_vector[2],
            1.0 - g_vector[1] - g_vector[2],
            g_vector[3],
            container,
        )
        / (
            likelihood(
                current_state[0],
                current_state[1],
                current_state[2],
                1.0 - current_state[1] - current_state[2],
                current_state[3],
                container,
            )
        )
    )
    if ratio >= 1:
        return g_vector
    if ratio < r:
        return current_state
    if ratio >= r:
        return g_vector


def MCMC(num_iter, container, include_systematic_errors=True, include_M_prior=False):
    """
    Run the Markov Chain Monte Carlo algorithm for num_iter steps on the likelihood distribution.
    """
    # create the random initial configuration in parameter space

    current_state = [
        np.random.normal(loc=74, scale=3),
        np.random.normal(loc=0.3, scale=0.0001),
        np.random.normal(loc=0.7, scale=0.0001),
        np.random.normal(loc=-19.23, scale=0.01),
    ]
    chain = [current_state]
    print("The first state is: " + str(current_state))
    print(
        "The initial Chi-Squared is: "
        + str(
            chi_squared(
                current_state[0],
                current_state[1],
                current_state[2],
                1 - current_state[1] - current_state[2],
                current_state[3],
                container,
            )
        )
    )
    for i in range(num_iter):
        link = metropolis(
            current_state,
            container,
            include_systematic_errors=include_systematic_errors,
            include_M_prior=include_M_prior,
        )
        chain.append(link)
        current_state = link.copy()
        if i % 1000 == 0:
            print(
                "The current state is: "
                + str(link)
                + "\t {0:.0%} completed".format(i / num_iter)
            )
            print(
                "The current Chi-Squared is: "
                + str(
                    chi_squared(
                        current_state[0],
                        current_state[1],
                        current_state[2],
                        1 - current_state[1] - current_state[2],
                        current_state[3],
                        container,
                    )
                )
            )

    # Don't include the beginning of the chain to ensure a steady state is shown.
    return np.array(chain[2000:])


class DataContainer(object):
    """
    A class used to import and store data of the form expected for this project
    
    Imports the parameter values from the file lcparam_DS17f.txt and stores them in 
    separate arrays, ordered by name (in this case name is an integer 0-39)
    
    lcparam_DS17f.txt is assumed to be stored in a directory called data that is 
    located in the same directory as BPE_functions.py
    
    Imports the systematic covariance matrix from the file sys_DS17f.txt 
    and stores it in a 2D array
    """

    def __init__(self):
        self.name = []
        self.z = []
        self.mb = []
        self.dmb = []
        self.systematic_covariance_matrix = []
        self.covariance_matrix = []
        self.inverted_covariance_matrix = []
        self.statistical_covariance_matrix = np.zeros(shape=(40, 40))
        self.inverted_statistical_covariance_matrix = []

    def import_data(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        param_filepath = dir_path + "/data/lcparam_DS17f.txt"
        data_filepath = dir_path + "/data/sys_DS17f.txt"

        self.name, self.z, self.mb, self.dmb = np.genfromtxt(
            param_filepath, usecols=(0, 1, 4, 5), delimiter=" ", unpack=True
        )

        covariance_values = []

        with open(data_filepath) as file:
            matrix_dimension = int(file.readline())
            for line in file:
                value = float(line)
                covariance_values.append(value)

        covariance_list = np.asarray(covariance_values)
        two_d_covariance_matrix = covariance_list.reshape(
            matrix_dimension, matrix_dimension
        )

        self.systematic_covariance_matrix = two_d_covariance_matrix
        self.covariance_matrix = np.copy(self.systematic_covariance_matrix)
        for i in range(40):
            self.covariance_matrix[i][i] += self.dmb[i] ** 2
            self.statistical_covariance_matrix[i][i] = self.dmb[i] ** 2

        self.inverted_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        self.inverted_statistical_covariance_matrix = np.linalg.inv(
            self.statistical_covariance_matrix
        )
