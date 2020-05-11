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

chain = bpe.MCMC(10000, c, include_systematic_errors=True, include_M_prior=True)

# Plotting function
fig = plt.figure()
ax = plt.axes()
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\Omega_\Lambda$")
plt.title(r"$oCDM$ Constraints For SN-only Sample")
corner.hist2d(chain[:, 1], chain[:, 2])
plt.show()


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


# Plot confidence intervals for 1-sigma and 2-sigma
omega_m = chain[:, 1]
omega_lambda = chain[:, 2]

cov = np.cov(omega_m, omega_lambda)
lambda_, v = np.linalg.eig(cov)
lambda_ = np.sqrt(lambda_)
ax = plt.subplot(111, aspect='equal')
plt.xlabel(r"$\Omega_m$")
plt.ylabel(r"$\Omega_\Lambda$")
plt.title(r"$1\sigma$" +  " and " + r"$2\sigma$" + " Confidence Intervals")
for j in range(1, 3):
    ell = Ellipse(xy=(np.mean(omega_m), np.mean(omega_lambda)),
                  width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell.set_facecolor('none')
    ell.set_edgecolor('black')
    ax.add_artist(ell)
plt.scatter(omega_m, omega_lambda)
plt.show()
