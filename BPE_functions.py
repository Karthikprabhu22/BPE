# Import this list of functions to run MCMC scripts with the S18 data.
import random
import numpy as np
import os
from numpy.random import default_rng
rng = default_rng()

def chi_squared(H_0, omega_m,omega_lam,omega_k):
    #TODO implement chi_squared function
    return


def likelihood(H_0, omega_m, omega_lam, omega_k):
    return np.exp(-chi_squared(H_0, omega_m,omega_lam,omega_k)/2)

def metropolis(current_state):
    """
    Perform one step of the metropolis algorithm, does not move time forward.
    The generating function is tbd.
    current_state[0]=H_0
    current_state[0]=Omega_m
    current_state[0]=Omega_lam
    current_state[3]=Omega_k
    """
    r = np.random.random()
    g_vector = [
        np.random.normal(current_state[0],scale=1.0),
        np.random.normal(current_state[1],scale=1.0),
        np.random.normal(current_state[2],scale=1.0),
        np.random.normal(current_state[3],scale=1.0)
    ]
    ratio = likelihood(g_vector[0], g_vector[1],g_vector[2],g_vector[3]) *
            prior(g_vector[0], g_vector[1],g_vector[2],g_vector[3]) /
            likelihood(current_state[0], current_state[1],current_state[2],current_state[3]) *
            prior(current_state[0], current_state[1],current_state[2],current_state[3])

    if ratio >= 1:
        return g_vector
    if ratio < r:
        return current_state
    if ratio > r:
        return g_vector

def prior():
    """
    For simplicity, we will use uniform prior for all the parameters
    
    Parameters
    ----------
    none
    
    Returns
    -------
    prior_vector: array
        An array of priors for all the 4 parameters
    """ 
    prior_H_0 = rng.uniform(67,73)
    prior_omega_m = rng.uniform(0.26,0.31)
    prior_omega_lam = rng.uniform(0.68,0.73)
    prior_M = rng.uniform(19.1,19.3)
    prior_vector = np.array([prior_H_0,prior_omega_m,prior_omega_lam,prior_M])
    return prior_vector

def MCMC(num_iter, likelihood, param_vector):
    """
    Run the Markov Chain Monte Carlo algorithm for num_iter steps on the likelihood distribution.
    """
    #create the initial configuration in parameter space
    current_state = [
        random.choice(param_vector[0]),
        random.choice(param_vector[1]),
        random.choice(param_vector[2]),
        random.choice(param_vector[3])
    ]
    chain = [current_state]
    for _ in range(num_iter):
        link = metropolis(current_state, likelihood)
        chain.append(link)
        current_state = link
    # Don't include the beginning of the chain to ensure a steady state.
    return chain[2000:]

class DataContainer(object):
    """
    A class used to import and store data of the form expected for this project
    """

    def __init__(self):
        self.name = []
        self.zcmb = []
        self.zhel = []
        self.dz = []
        self.mb = []
        self.dmb = []
        self.x1 = []
        self.dx1 = []
        self.color = []
        self.dcolor = []
        self.covariance_matrix = []

    def import_params(self):
        """
        Imports the parameter values from the file lcparam_DS17f.txt and stores them in separate arrays, ordered by name (in this case name is an integer 0-39)
        lcparam_DS17f.txt is assumed to be stored in a directory called data that is located in the same directory as BPE_function.py
        """ 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath =  dir_path + "/data/lcparam_DS17f.txt"
        
        self.name, self.zcmb, self.zhel, self.dz, self.mb, self.dmb, self.x1, self.dx1, self.color, self.dcolor = np.genfromtxt(filepath, usecols=(0,1,2,3,4,5,6,7,8,9), delimiter =' ', unpack = True)

    def import_covariance_matrix(self):
        """
        Imports the systematic covariance matrix from the file sys_DS17f.txt and stores it in a 2D array
        """

        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath =  dir_path + "/data/sys_DS17f.txt"

        covariance_values = []

        with open(filepath) as file:
            matrix_dimension = int(file.readline())
            for line in file:
                value = float(line)
                covariance_values.append(value)
        
        covariance_matrix = np.asarray(covariance_values)
        covariance_matrix.reshape(matrix_dimension, matrix_dimension)

        self.covariance_matrix = covariance_matrix

def Ez(z,omega_m,omega_lam,omega_k):
    return np.sqrt(omega_m*(1+z)**3+omega_lam+omega_k*(1+z)**2)

def integrate_Ez_prime(z,num_bins,omega_m,omega_lam,omega_k):
    step_size=z/num_bins
    Total=0
    z_prime=0

    for _ in range(len(num_bins)):
        Total+= (Ez(z_prime+step_size,omega_m,omega_lam,omega_k)+Ez(z_prime+step_size,omega_m,omega_lam,omega_k))*step_size/2
        z_prime+=step_size
    return Total


def mu_data(m_B,M):
    """
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