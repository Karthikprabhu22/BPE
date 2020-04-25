# Import this list of functions to run MCMC scripts with the S18 data.
import random
import numpy as np
import os


def metropolis(current_state, posterior, param_vector):
    """
    Perform one step of the metropolis algorithm, does not move time forward.
    The generating function is tbd.
    """
    r = np.random.random()
    g_vector = [
        random.choice(param_vector[0]),
        random.choice(param_vector[1]),
        random.choice(param_vector[2]),
        random.choice(param_vector[3]),
    ]
    ratio = posterior[g_vector] / posterior[current_state]
    if ratio >= 1:
        return g_vector
    if ratio < r:
        return current_state
    if ratio > r:
        return g_vector


def MCMC(num_iter, posterior, param_vector):
    """
    Run the Markov Chain Monte Carlo algorithm for num_iter steps on the posterior distribution.
    """
    current_state = [
        random.choice(param_vector[0]),
        random.choice(param_vector[1]),
        random.choice(param_vector[2]),
        random.choice(param_vector[3]),
    ]
    chain = [current_state]
    for _ in range(num_iter):
        link = metropolis(current_state, posterior, param_vector)
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







