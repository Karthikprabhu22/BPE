# Import this list of functions to run MCMC scripts with the S18 data.
import random
import numpy as np


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
