# Import this list of functions to run MCMC scripts with the S18 data.
import random
import numpy as np


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

def prior(H_0, omega_m, omega_lam, omega_k):
    #TODO implement prior function
    return

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



