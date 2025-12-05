---
title: Paper on photometric redshifts with latent SEDs and Gaussian Processes
author: admin
guid:
desc:
permalink: /2016-12-02-paper-on-photoz-via-gaussian-processes/
layout: post
---

David Hogg and I finished our first paper about photometric redshifts: [https://arxiv.org/abs/1612.00847](https://arxiv.org/abs/1612.00847). I say first, because it is part of an effort to design a new approach to construct and calibrate a likelihood function to estimate photometric redshifts from complicated photometric data.

Our code is publicly available, and documented: [https://github.com/ixkael/Delight](https://github.com/ixkael/Delight).

In this paper, we extend template fitting in an atypical way: instead of relying of a (small) set of synthetic templates to compute photo-z's, like standard template fitting approaches (e.g., BPZ), we construct a probabilistic template (i.e., a template flux-redshift model with error bars) for each galaxy of our training set.
To estimate the redshift of a target galaxy, we compare it to each training galaxy, or, more precisely, to the probabilistic template made with each target galaxy.
The posterior distribution, thanks to a discrete version of a type marginalization in Bayes theorem, is just a linear sum over training galaxies.

One particularly interesting feature of this work is the ability to quickly create a probabilistic SED model with each training galaxy, for which we only have deep photometry and a (good) redshift.
We don't fit for the data in wavelength space in a brute force fashion, which could be obtained by integrating each SED (for example those created in a MCMC run) in photometric bands to generate the noiseless photometric fluxes and compare to our noisy observed fluxes.
Instead, we directly fit for the fluxes - we simply constrain this fit to correspond to an SED in wavelength space.
The reason this is at all possible is because the integration of an SED in a photometric filter is a linear operation.
Oh, and we use the magic of Gaussian Processes.
By assuming that the SED is described by a linear mixture of galaxy templates with Gaussian Process residuals, the fluxes as a function of photometric band and redshift are also a Gaussian Process! The mean function and kernel have nasty but closed mathematical form.
This means we can really fit for the multi-band fluxes of a galaxy and impose that the fit must correspond to an SED (with some covariance) without explicitly going to wavelength space.
This makes the fit extremely easy and fast, and we can create a latent SED model, and probabilistic templates as a function of band and redshift, for all our training data (i.e., thousands to hundreds of thousands of galaxies).

We are actively working on extending this method to not require spectroscopic redshifts at all, and to also exploit the known low-dimensionality of galaxy spectra. Oh, and adding dust and other physical parameters of interest. Those three points are the main drawbacks of the method but can be resolved.

As a side note, I am also excited by this approach because it provides a great tool to quickly fit supernova lightcurves, with a latent SED model and natural way to compare lightcurves, and maybe perform classification and regression.
And it can also be used as a generative model to draw new objects (fluxes and redshifts) that are statistically consistent with a training set. This could be a way to populate simulations with realistic galaxies (with realistic fluxes, redshifts and SEDS).
