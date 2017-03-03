---
id: 230
title: Crash course on the Dirichlet and Multinomial distributions as conjugate priors
author: admin
guid:
desc:
permalink: /dirichlet-multinomial-conjugate-priors/
layout: post
---

*This is a concise introduction to the Dirichlet distribution and its connection to Multinomial-distributions counts, based on work and discussions with [Daniel Mortlock](http://astro.ic.ac.uk/dmortlock/home) (Imperial College), [Hiranya Peiris](http://zuserver2.star.ucl.ac.uk/~hiranya/) (UCL), [Josh Speagle](https://joshspeagle.github.io/) (Harvard), and [Jo Bovy](http://astro.utoronto.ca/~bovy/) (Toronto).
Everything below is very simple but for some reason misunderstood and massively under-exploited outside of the statistics literature.*

The **Dirichlet distribution** is a very useful tool in probability and statistics. It is a natural choice for constructing a prior on a set of $$K$$ positive unit-normalized weights, denoted by $$\vec{f}=\{f_i \vert  i=1, \cdots, K\}$$ with $$f_i > 0\ \forall i$$ and $$\sum_if_i = 1$$.

To use a Dirichlet prior for those weights, one needs a set of (positive, real) *hyperparameters* $$\vec{\alpha}=\{\alpha_i \vert  \alpha_i i=1, \cdots, K\}$$ with $$\alpha_i > 0 \ \forall i$$. Neglecting the normalization (i.e. the terms which do not depend on $$\vec{f}$$, which you can find [here](https://en.wikipedia.org/wiki/Dirichlet_distribution)), the prior reads

$$
\vec{f} \vert  \vec{\alpha} \sim \mathrm{Dir}(\vec{\alpha}) \quad \Leftrightarrow \quad p(\vec{f} \vert  \vec{\alpha})\propto \prod_i f_i^{\alpha_i-1}
$$.

Note that this is a prior on the simplex, and that it is easy to draw values (weights $$\vec{f}$$) from this distribution, like you would do from uniform or Gaussian distributions. By changing the hyperparameters, one can affect the statistical properties of the weights. For example, if we write $$A = \sum_i \alpha_i$$, their mean, variance and covariance read

$$
\mathbb{E}[f_i] = \alpha_i/A  \quad  \mathrm{Var}[f_i] = \alpha_i (A-\alpha_i) / A^2 (A+1) \quad  \mathrm{Cov}[f_i f_j] = - \alpha_i \alpha_j / A^2 (A+1)
$$,

where you will forgive me for slightly abusing the notation for expectation values.
You will note that the coefficients are anti-correlated. *Can you find why this is intuitively correct?*
An important case is $$\alpha_i = 1 \ \forall i$$, where the mean becomes $$1/K$$.
This is the most general, *uninformative prior* you can place on a set of unit-normalized positive weights, and it has important consequences.

One of those has to do with what these sorts of weights are used for: drawing objects from categories. Indeed, $$\vec{f}$$ can be interpreted as a set of probabilities over $$K$$ classes. Drawing objects from those classes (one by one) is known as a [categorical](https://en.wikipedia.org/wiki/Categorical_distribution) draw. The generalization of these draws is the well-known [Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution), which corresponds to drawing $$N$$ objects from $$K$$ categories/classes with probabilities $$\vec{f}$$. We are interested in the probability of drawing some the numbers of objects the classes, $$\vec{n}=\{n_i \vert  i=1, \cdots, K\}$$, with $$\sum_i n_i = N$$. The Multinomial distribution is

$$
\vec{n} \vert  \vec{f}, N \sim \textrm{Mult}(\vec{f}) \quad \Leftrightarrow \quad p(\vec{n} \vert  \vec{f}, N)\propto \prod_i f_i^{n_i}
$$.

where I have, again, neglected the normalization constants that do not depend on $$\vec{n}$$. Note that we have conditioned on $$N$$ here (things get *much* more complicated if the number of objects varies!).

Did you notice how similar this expression is to the Dirichlet distribution? This is because the Dirichlet and Multinomial distributions are **conjugate priors**. If you swap the parameters $$\vec{n}$$ and $$\vec{f}$$ in the probability $$p(\dot \vert   \dot)$$, you change one distribution into the other. This is extremely useful, because this sort of swap happens when applying **Bayes theorem**:

$$
p(A\vert B) = p(B\vert A)p(A)/p(B) \propto p(B\vert A)p(A)
$$.

**Let us take a concrete, classic example**: I observe a set of categorical counts $$\vec{n}$$, and I want to estimate the weights $$\vec{f}$$, subject to a Dirichlet prior controlled by fixed hyperparameters $$\vec{\alpha}$$. Applying Bayes theorem gives

$$
p(\vec{f}\vert \vec{n}, \alpha, N) \propto p(\vec{n} \vert  \vec{f}, N) p(\vec{f} \vert  \vec{\alpha}) \propto   \prod_i f_i^{n_i+\alpha_i-1}
$$,

**which is a Dirichlet distribution**! So we can write $$\vec{f} \vert  \vec{n}, \vec{\alpha}  \sim \mathrm{Dir}(\vec{n} + \vec{\alpha})$$.

With the uninformative prior $$\alpha_i =1 \ \forall i$$, we have $$\vec{f} \vert  \vec{n}  \sim \mathrm{Dir}(\vec{n} + 1)$$ which is a common example. This is the posterior distribution for the probabilities of the $$K$$ categories given the observed counts $$\vec{n}$$. The weights could be the normalized amplitudes of a histogram, for example. The reason this is a powerful result is because we know how to sample directly from a Dirichlet distribution, and we can easily compute all its moments; there is no need to resort to MCMC or other sampling techniques to explore or approximate this posterior distribution. Note that this is not quite true if the number counts themselves are unobserved (latent) variables in a larger hierarchical probabilistic model, but one can still exploit the Dirichlet-Multinomial conjugacy property via Gibbs MCMC sampling, like we did [in this paper](https://arxiv.org/abs/1602.05960) for galaxy redshift estimation.

One final remark: an other very common structure is the so-called **Dirichlet-Multinomial distribution or compound**. This arises when one wants to skip the weights $$\vec{f}$$ and directly connect the number counts $$\vec{n}$$ to the hyperparameters $$\vec{\alpha}$$. This requires marginalizing out the weights $$\vec{f}$$, i.e.,

$$
p(\vec{n} \vert  \vec{\alpha}, N) = \int d\vec{f} p(\vec{n} \vert  \vec{f}, N) p(\vec{f} \vert  \vec{\alpha})
$$

This admits a closed form which you can find [here](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution). It is a useful when one wants is not interested in the weights $$\vec{f}$$, or if one wants to infer a point estimate of $$\vec{\alpha}$$ (optimal in some sense, such as a maximum likelihood solution) given a set of number counts $$\vec{n}$$ while marginalizing over the weights.

$$
p(\vec{\alpha} \vert  \vec{n}, N) \propto p(\vec{n} \vert  \vec{\alpha}, N) p(\vec{\alpha})
$$

which involves the Dirichlet-Multinomial compound and a prior over $$\vec{\alpha}$$.
