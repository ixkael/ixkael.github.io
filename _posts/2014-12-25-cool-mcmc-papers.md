---
id: 160
title: Cool MCMC papers
date: 2014-12-25T17:35:48+00:00
author: admin
layout: post
guid: http://ixkael.com/blog/?p=160
permalink: /cool-mcmc-papers/
categories:
  - News, Papers, Press
---
I've been reading a lot about Bayesian inference and MCMC lately. Not about the generalities, which I have been using for years, but rather about the technicalities of advanced algorithms. I recently realised that the tools that we cosmologists use are very basic, and sometimes far from the actual state of the art in statistical inference. Another reason is that I have started a couple of projects that require more advanced tools and tricks to become tractable. Anyway, here is a few papers that I found clear and detailed, while intelligible by physicists I think. I will update this post as I read more.

<!--more-->

## Hamiltonian Monte Carlo (HMC)

_HMC is a powerful MCMC method to explore complicated distributions with high acceptance rates and a lot of technical and computational flexibility._ 

[**MCMC using Hamiltonian dynamics** by Radford M. Neal.](http://arxiv.org/abs/1206.1901) A very progressive and pedagogic introduction to HMC techniques, their variants and latest developments. I particularly liked the clear presentation of the pros and cons of HMC illustrated on simple examples. Bonus point: it is a goldmine of references.

## Sequential Monte Carlo (SMC) and Particle Filtering

_SMC is a useful category of iterative algorithms for MCMC and optimisation, especially for Gaussian state space models and Bayesian networks._

[**An introduction to Sequential Monte Carlo methods** by A. Doucet, N. de Freitas, and N. Gordon.](http://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.pdf) A very short (and slightly outdated) but simple introduction to SMC &#8211; simpler than other tutorial papers I find.

[**A Tutorial on Particle Filtering and Smoothing:Fifteen years later** by A. Doucet and A. M. Johansen.](http://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf) As the title indicates, this one is a proper tutorial, detailing the state of the art of SMC in 2008. The emphasis is very much on the mathematics so the barrier to entry may be quite high for astronomers and physicists without an application at hand. But it is an excellent exhaustive tutorial.

[**Particle Methods: an introduction with applications** by P. Del Moral and A. Doucet.](http://www.esaim-proc.org/articles/proc/pdf/2014/01/proc144401.pdf) An even more mathematical paper but very clear and progressive.

## Other (more generic) resources

[**Information Theory, Inference, and Learning Algorithms** by David J.C. MacKay.](http://www.inference.phy.cam.ac.uk/itprnn/book.pdf) Obviously one of the best sources of information on these topics.