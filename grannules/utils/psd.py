r"""
Red Giant Power Spectrum Utilities
**********************************

Implements helper methods to generate synthetic power spectrum envelopes without
white noise or activity similar to `de Assis Peralta et al. 2018`_ Equation
(9).

.. _de Assis Peralta et al. 2018: https://doi.org/10.48550/arXiv.1805.04296

.. math::
    \mathrm{PSD}(\nu) = 
    \eta^2(\nu) 
    \left[ 
        \frac{P}{1 + \left(2 \pi \tau \nu \right)^\alpha} 
        + H \exp \left(
          \frac{-(\nu - \nu_\mathrm{max})^2}
          {0.66 \nu_\mathrm{max}^{0.88} /4 \ln 2}
        \right)
    \right]

The whole equation is implemented in :meth:`PSD`, and individual components are
implemented as follows: 

* The damping factor, :math:`\eta^2(\nu)` is implemented in :meth:`damping`
* The granulation background, the first term in the brackets, is implemented in
  :meth:`granulation`
* The signal excess envelope, the second term in the brackets, is implemented in
  :meth:`excess`.

This submodule also implements simple scaling relations for
:math:`\nu_\mathrm{max}`, the frequency at maximum power, and
:math:`\Delta \nu`, the large frequency separation in :meth:`nu_max` and
:meth:`delta_nu` respectively.
"""

import numpy as np
import pandas as pd

from .scalingrelations import SRPredictor

NYQUIST = 283.2114

def PSD(nu, nu_max, H, P, tau, alpha, reshape = True):
    """
    :param reshape: Reshapes nu into a "row" array of shape (1, len(nu)) and the
        other input parameters into "column" arrays of shape (len(arg), 1) such
        that the output is a 2D array where the ijth element corresponds to
        power of the ith star at the jth frequency. Defaults to True.
    :type reshape: bool
    """

    if reshape:
        nu = np.reshape(nu, (1, -1))

        nu_max = np.reshape(nu_max, (-1, 1))
        H = np.reshape(H, (-1, 1))
        P = np.reshape(P,(-1, 1))
        tau = np.reshape(tau, (-1, 1))
        alpha = np.reshape(alpha, (-1, 1))
    
    eta = damping(nu)
    b = granulation(nu, P, tau, alpha)
    g = excess(nu, nu_max, H)
    return eta * (b + g)

# PSD model per deassis
def damping(nu):
    eta = np.sinc((1 / 2) * (nu / NYQUIST)) ** 2
    return eta

def granulation(nu, P, tau, alpha):
    granulation = (P / (1 + (2 * np.pi * tau * 1e-6 * nu) ** alpha))
    if not np.isfinite(granulation).all():
        return nu * np.inf
    return granulation

def excess(nu, nu_max, H):
    FWHM = 0.66 * nu_max ** 0.88
    G = H * np.exp(-(nu - nu_max)**2 / (FWHM ** 2 / (4 * np.log(2))))
    return G

def nu_max(M, R, Teff, nu_max_solar = 3090, Teff_solar = 5777):
    return nu_max_solar * M * (R **-2) * ((Teff / Teff_solar)**-0.5)

def delta_nu(M, R, delta_nu_solar = 135.1):
    return delta_nu_solar * (M ** 0.5) * (R ** -1.5)