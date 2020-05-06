"""
Use this function to run the Bayesian Parameter Estimation to infer the values of cosmological parameters from the Supernova dataset.
"""
import BPE_functions as bpe
import numpy as np
import matplotlib.pyplot as plt

c = bpe.DataContainer()

c.import_data()

chain = bpe.MCMC(10000, c)

# Put plots here

