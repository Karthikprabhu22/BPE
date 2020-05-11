"""
Use this function to run the Bayesian Parameter Estimation to infer the values of cosmological parameters from the Supernova dataset.
"""
import BPE_functions as bpe

import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.patches import Ellipse


c = bpe.DataContainer()
c.import_data()

chain = bpe.MCMC(30000, c, include_systematic_errors=True, include_M_prior=False)

# Plotting function
fig = plt.figure()
ax = plt.axes()
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\Omega_\Lambda$")
plt.title(r"$oCDM$ Constraints For SN-only Sample")
corner.hist2d(chain[:, 1], chain[:, 2])
plt.show()
plt.savefig("omol.png")


# Plot the other fig
mu_D = c.mb - chain[:, 3].mean()
mu_T = bpe.signal(
    chain[:, 0].mean(),
    chain[:, 1].mean(),
    chain[:, 2].mean(),
    1 - chain[:, 1].mean() - chain[:, 2].mean(),
    c.z,
)
fig = plt.figure()
ax = plt.axes()
plt.subplot(211)
plt.xlabel(r"$z$")
plt.ylabel("Distance Modulus (mag)")
plt.semilogx()
plt.plot(c.z, mu_D)
plt.plot(c.z, mu_T, "o")

plt.subplot(212)
plt.ylabel("Hubble Res(mag)")
plt.semilogx()

plt.plot(c.z, [0] * 40)
plt.errorbar(c.z, mu_D - mu_T, fmt="o", yerr=c.dmb)
plt.show()
plt.savefig("lum_dist.png")


# Plot the other fig
fig = plt.figure()
plt.xlabel("$H_0$")
plt.ylabel("Posterior")
plt.show()
plt.savefig(r"Posterior distribution of $H_0$.png")
