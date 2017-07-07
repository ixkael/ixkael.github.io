---
id: 241
title: Does adding an unconstrained parameter change the Bayesian evidence?
author: admin
guid:
desc:  
permalink: /does-adding-an-unconstrained-parameter-to-a-model-change-the-bayesian-evidence/
layout: post
---

_Thanks to Niall MacCrann (OSU) and Joe Zuntz (ROE) for triggering this discussion and pushing me to finally put this quick note online._
 
Here is a simple premise: we constrain the parameters of a model, for example by inferring their posterior distribution, and then consider extensions of this model, add parameters, and repeat the analysis. 
One obvious way to compare models in the Bayesian framework is to use _Bayes factors_, which involve _evidence ratios_.

In some cases, some of the extra parameters may be __unconstrained__. 
_Do those unconstrained parameters affect the Bayesian evidence?_ 

It is not trivial to answer this question by looking at the generic formulation of Bayes factors. But it can be addressed by using the __Savage-Dickey Density Ratio (SDDR)__, which we will re-derive here.

Let us consider a base model $$M_1$$ consisting of a set of parameters $$\theta$$ that we constrain with data $$D$$. 
The Bayesian evidence is
$$E_1 = p(D\vert M_1) = \int \mathrm{d}\theta p(D\vert \theta, M_1) p(\theta \vert  M_1)$$
where the two distributions in the integral are the likelihood and the prior, respectively.

We now consider a second model $$M_2$$, which extends $$M_1$$ in the sense that it includes the parameters $$\theta$$ as well as other parameters $$\phi$$. Importantly, $$M_2$$ reduces to $$M_1$$ when $$\phi=\phi^*$$, and we say that the models are nested.

We can write the Bayesian evidence as 
$$E_2 = p(D\vert M_2) = \int \mathrm{d}\theta \mathrm{d}\phi  p(D\vert \theta, \phi, M_2) p(\theta, \phi \vert  M_2)$$
where we have introduced the likelihood and the prior for $$M_2$$.
Let's also rewrite the evidence as
$$E_2 = \frac{ \int \mathrm{d}\theta \mathrm{d}\phi  p(D\vert \theta, \phi, M_2) p(\theta, \phi \vert  M_2) }{ p(\phi \vert  D, M_2)}$$
where we now make use of a _marginal posterior distribution_, where $$\theta$$ has been marginalized over, but not $$\phi$$. Importantly, this is valid for any value of $$\phi$$, and in particular $$\phi^*$$.

Our final ingredient is to connect the likelihoods in the two models by noting that 
$$p(D\vert \theta, M_1)  = p(D\vert \theta, \phi=\phi^*, M_2)$$.

The evidence ratio between the two models (connected to the Bayes factor via a multiplicative term $$p(M_1)/p(M_2)$$) reads
$$\frac{E_1}{E_2} = p(\phi^* \vert  D, M_2) \frac{\int \mathrm{d}\theta p(D\vert \theta, \phi^*, M_2) p(\theta \vert  M_1)}{\int \mathrm{d}\theta \mathrm{d}\phi  p(D\vert \theta, \phi^*, M_2) p(\theta, \phi^* \vert  M_2)}$$.

We immediately see that if the priors on $$\theta$$ and $$\phi$$ are independent and $$p(\theta\vert M_1) = p(\theta\vert M_2)$$, which is a very common situation, the Bayes factor simplifies to

$$\frac{E_1}{E_2} = \frac{p(\phi^* \vert  D, M_2)}{p(\phi^* \vert  M_2)}$$

In other words the ratio between the marginalized posterior and prior on $$\phi$$ evaluated at $$\phi^*$$ in the extended model $$M_2$$.
This is the famous Savage-Dickey Density Ratio (SDDR). 
Another pedagogical derivation can be found in [this paper](https://arxiv.org/abs/1307.2904), and some useful remarks at [here](https://xianblog.wordpress.com/2009/09/24/is-the-dickey-savage-ratio-any-valid/).

__We now go back to our original question: we add one or multiple parameters $$\phi$$ to a model $$M_1$$, with prior $$p(\phi \vert  M_2)$$. If those parameters come out truly unconstrained by our data, then the marginalized posterior distribution $$p(\phi \vert  D, M_2)$$ will reduce to the prior, regardless of $$\phi^*$$. Hence, the evidence ratio $$E_1/E_2$$ is equal to one, and _this parameter has no effect on our inference or interpretation of the data_. 
