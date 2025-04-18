grannules
=========

.. raw:: html

   <div style="display: inline-block;">
      <a href="https://pypi.python.org/pypi/grannules">
         <img src="https://img.shields.io/pypi/v/grannules.svg" alt="PyPI version" />
      </a>
      <!--
      <a href="https://github.com/andres-luengo/grannules/blob/main/LICENSE">
         <img src="https://img.shields.io/badge/license-MIT-orange.svg?style=flat" alt="MIT License" />
      </a> 
      -->
   </div>

A package for predicting granulation light variability parameters of red giant
stars.

Red giant power spectra generally look like Figure 4 of de Assis Peralta et al.
2018 [1]_:

.. image:: ../images/deassisgraph.png
   :width: 400
   :alt: Red giant power spectrum with model components overplotted.

The blue curve represents the component of the spectrum originating from the
granulation of the star. In other words, the light variability caused by the
appearance and disappearance of "granules," convection cells on the star's
photosphere. The red curve represents the component of the spectrum originating
from stellar oscillations.

Red giant asteroseismologists usually focus on the signal received from the
star, (the spikes of the red curve) whose properties have tight relations
with the physical properties of the star. However, understanding the background
components of these power spectra could also prove useful. 

The granulation component is described by

.. math::

   G(\nu) = \frac{P}{1 + (2 \pi \tau \nu)^\alpha}

and the overall shape of the oscillations component (ignoring the signal spikes)
by a Gaussian function of the following form

.. math::

   O(\nu) = H \exp \left[ \frac{- (\nu - \nu_\mathrm{max})^2}{\delta 
   \nu_\mathrm{env}^2 / 4 \ln 2} \right]

where :math:`\tau` is the timescale of granulation, and
:math:`\delta \nu_\mathrm{env}` is the FWHM of the gaussian, which we take to be
:math:`\delta \nu_\mathrm{env} = 0.66 \nu_\mathrm{max}^{0.88}` [1]_, [2]_.
:math:`H,\, P,\, \tau,\,` and :math:`\alpha` are usually fitted parameters.
There exist scaling relations for these values, but they are not accurate enough
for practical use [1]_. 

This is where ``grannules`` comes in. This Python package is essentially a
wrapper around a neural network that, given a star's mass, radius, temperature,
luminosity, and evolutionary phase, can predict that star's background
parameters much more accurately than the existing scaling relations.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   neural_net
   utils
   usage
   webapp
   contributing
   authors
   history

.. [1] `de Assis Peralta, R., et al. 2018 <https://doi.org/10.48550/arXiv.1805.04296>`_

.. [2] `Mosser et al. 2012 <https://doi.org/10.1051/0004-6361/201117352>`_