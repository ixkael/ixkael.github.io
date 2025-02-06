---
id: 669
title: Lutz-Kelker correction for low SNR parallax
date: 2016-10-01T14:54:57+00:00
author: admin
layout: post
guid: http://ixkael.com/blog/?p=669
permalink: /lutz-kelker-correction-for-low-snr-parallax/
categories:
  - News, Papers, Press
---
As advertised in [David Hogg's recent tweets](https://twitter.com/davidwhogg/status/782193328318185473), Hogg and I re-derived a [Lutz-Kelker correction](http://adsabs.harvard.edu/abs/1973PASP...85..573L) for low SNR parallax during one of our recent group meetings (now held at the Simons Center for Computational Astronomy in NYC). I am including the derivation here for my records. The idea is the following: standard parallax measurements (estimates and their Gaussian errors) can be improved by including prior information. Specifically, the prior for distances in 3D space is $$p(d) = d^2$$, and we aim to compute a maximum a posteriori estimate of the parallax given the initial estimate, its error, and the prior.

The full posterior distribution given the parallax estimate and its error is

$$
p(\varpi|\hat{\varpi}, \sigma_{\hat{\varpi}}) = p(\hat{\varpi}, \sigma_{\hat{\varpi}} | \varpi) p(\varpi)

$$

Finding the maximum of this distribution with a uniform prior would give us the initial estimate. But let's use the improved parallax prior

$$
p(\varpi) = p(d) |\frac{\partial \varpi}{\partial d}| = \varpi^{-4}
$$

In this case, taking equating the derivate of the posterior distribution leads to the maximum (a posteriori) estimate

$$
\hat{\varpi}^\mathrm{MAP} = \hat{\varpi} \left(\frac{1}{2} + \frac{1}{2}\sqrt{ 1-\left(4\frac{\hat{\varpi}}{\sigma_{\hat{\varpi}}}\right)^2 } \right) = \hat{\varpi} \left(\frac{1}{2} + \frac{1}{2}\sqrt{ 1-16\ \mathrm{SNR}^2 } \right)

$$
