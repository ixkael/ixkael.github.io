---
title: Paper on hierarchical color--magnitude diagram of Gaia stars
author: admin
guid:
desc:
permalink: /2017-03-23-paper-on-hierarchical-cmd/
layout: post
---

We finally put out our paper on a hierarchical inference of stellar distances and color--magnitude diagram of Gaia stars: [https://arxiv.org/abs/1703.08112](https://arxiv.org/abs/1703.08112).

Our code is all public: [https://github.com/ixkael/Starlight](https://github.com/ixkael/Starlight). Unfortunately, it is not very well documented yet. But if you complain to me via email or twitter I will complete the documentation!

The idea originated during the Gaia Sprint hosted last September at the Flatiron Institute in NYC.
This is a nice proof of concept that we can constrain the color--magnitude diagram of stars directly from noisy colors and parallaxes, from a data set like Gaia (which we use in this work).
As a consequence of Bayesian shrinkage (a typical feature of hierarchical models), the distances of individual objects are improved when one considers them together (as opposed to individually) and infers their distribution.
One nice aspect of this analysis (aside from [some technical tricks](https://astrostatistics.wordpress.com/2017/04/04/a-convexified-mixture-model/)) is that it doesn't use stellar models - it only uses the data at hand.
We model the color--magnitude diagram with a linear mixture of Gaussians.
It is a proof of concept because we would like to add many components to this model in order to have a more complete physical model of the gaia data: we will ultimately infer the spatial distribution of stars and dust, as well as the selection function of the data.
Some of those elements will hopefully be implemented during the next Gaia Sprint, hosted in Heidelberg this summer!

A great companion paper lead by Lauren Anderson and David Hogg will come out very soon and showcase another aspect of this sort of hierarchical analysis of the Gaia data.
