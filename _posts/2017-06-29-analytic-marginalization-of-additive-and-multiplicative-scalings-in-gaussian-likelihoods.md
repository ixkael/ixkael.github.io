---
id: 240
title: Analytic marginalization of additive and multiplicative scalings in gaussian likelihoods
author: admin
guid:
desc:  
permalink: /analytic-marginalization-of-additive-and-multiplicative-scalings-in-gaussian-likelihoods/
layout: post
---
 
Let us consider some data $$\vec{y}$$ (a vector of dimension $$D$$) and try to explain them with a model $$\vec{x}$$, which may be parameterized in some way (for the purpose of this post we don't need to write it explicitly). 
We allow for an additive and a multiplicative scaling of the model, described by two scalar parameters $$a$$ and $$s$$, so that the full model with fit $$\vec{y}$$ with is $$a+s\vec{x}$$.

We make the strong  assumption that our data uncertainties are Gaussian and described with a $$D\times D$$ covariance matrix $$\Sigma_y$$. As a result, the likelihood function is

 $$ p(\vec{y}|\vec{x}, \Sigma_y, s, a) = \mathcal{N}(a+s\vec{x} - \vec{y}; \Sigma_y) $$
 
 We will also adopt Gaussian priors on the additive and multiplicative scaling parameters:
 
 $$ p(s) = \mathcal{N}(s-\hat{s};\sigma^2_s) $$ and $$p(a) = \mathcal{N}(a-\hat{a};\sigma^2_a) $$
 
 Whether those are broad or narrow is irrelevant; all that matters is that $$s$$ and $$a$$ are real numbers wich may be constrained by prior or external information.
 
 In any Bayesian analysis of our data $$\vec{y}$$ (hierarchical or not), we will have to compute terms like 
 
 $$L = p(\vec{y}|\vec{x}, \Sigma, s, a)p(s)p(a) =  \mathcal{N}(s-\hat{s};\sigma^2_s)  \mathcal{N}(a+s\vec{x} - \vec{y}; \Sigma_y) \mathcal{N}(a-\hat{a};\sigma^2_a)  $$
 
 which will show up as soon as we try to infer the parameters in $$\vec{x}$$ while correctly dealing with $$s$$ and $$a$$. 
 The typical approach to tackle this problem is to perform parameter inference (e.g., via MCMC sampling) for all the parameters, including $$a$$ and $$s$$. However, given the simplicity of those two, we might wonder if we could get rid of them analytically. This is especially relevant if we don't particularly care about them and they are nuisance parameters allowing us to fit $$\vec{y}$$ better (i.e., we are only truly interested in the parameters of $$\vec{x}$$.  
 

 We will make use of the following identity and analytic marginalization:
 
 $$\int \mathrm{d}x \mathcal{N}(x-a;A)\mathcal{N}(x-b;B) \\ = \int \mathrm{d}x \mathcal{N}(a-b;A+B)\mathcal{N}(x-(A^{-1}+B^{-1})^{-1}(A^{-1}a+B^{-1}b);(A^{-1}+B^{-1})^{-1}) \\ = \mathcal{N}(a-b;A+B)$$ 
 

This allows us to perform a first simplification in $$a$$:

$$\mathcal{N}(a+s\vec{x} - \vec{y}; \Sigma_y) \mathcal{N}(a-\hat{a};\sigma^2_a)  \\ = \mathcal{N}(s\vec{x} + \underbrace{\hat{a} - \vec{y}}_{\vec{y}_a}; \underbrace{\Sigma_y + \sigma^2_a I_D}_{\Sigma_{ya}}) \\ \times \mathcal{N}(a-\underbrace{(\sigma^{-2}_a I_D+\Sigma_y^{-1})^{-1}(\sigma^{-2}_a I_D\hat{a}+\Sigma_y^{-1}(\vec{y}-s\vec{x}))}_{a^{\mathrm{MAP}}};\dots) $$

We see that this distribution is Gaussian in $$a$$, and the maximum a posteriori value is $$a^{\mathrm{MAP}}$$. I didn't write the covariance in $$a$$ due to space constraints, but it is easy to derive. And as I will discuss below, we do not need it if we want to marginalize over $$a$$. I have also introduced $$\vec{y}_a$$ and $$\Sigma_{ya}$$ to shorten the equations below.

The second simplification, over $$s$$ this time, is slightly less trivial, but leads us to something like
 
 $$\mathcal{N}(s-\hat{s};\sigma^2_s)  \mathcal{N}(s\vec{x} + \vec{y}_a; \Sigma_{ya}) \\ = \Bigl( (2\pi)^D \ F_\mathrm{TT}  \ \sigma^2_s \ |\Sigma_{ya}|  \Bigr)^{-1/2} \exp\left( - \frac{1}{2} F_\mathrm{OO}   + \frac{1}{2} \frac{F_\mathrm{OT} ^2}{F_\mathrm{TT} } \right) \ \mathcal{N}\Bigl(s- \underbrace{\frac{F_\mathrm{OT} }{F_\mathrm{TT} }}_{s^\mathrm{MAP}}; \dots\Bigr)$$
 
 with the terms
 
 $$F_\mathrm{OO} = \vec{y}_a \  \Sigma_{ya}^{-1}\ \vec{y}_a \ +\  \hat{s}^2/\sigma^2_s $$
 
 $$F_\mathrm{TT}  = \vec{x}^T\  \Sigma_{ya}^{-1}\ \vec{x} \ + \ 1/\sigma^2_s$$
 
 $$F_\mathrm{OT}  = \vec{x}^T \ \Sigma_{ya}^{-1}\ \vec{y}_a \ + \ \hat{s}/\sigma^2_s$$
 
 Again, we see that this distribution is Gaussian in $$s$$, and the maximum a posteriori value is $$s^{\mathrm{MAP}}$$. Similarly to $$a$$, I didn't write the covariance in $$s$$ but it is easy to derive and it is not needed if we want to marginalize over $$s$$.
 
 What does it tell us? We have re-written our target distribution $$L$$ as
 
 $$\mathcal{N}(s-{s}^\mathrm{MAP}; \dots) \ \mathcal{N}(a-{a}^\mathrm{MAP}; \dots) \ \Bigl( (2\pi)^D  F_\mathrm{TT}   \sigma^2_s \ |\Sigma_{ya}|  \Bigr)^{-1/2} \exp\left( - \frac{1}{2} F_\mathrm{OO}   + \frac{1}{2} \frac{F_\mathrm{OT} ^2}{F_\mathrm{TT} } \right)  $$
 

 This is great, because we have two elegant solutions to deal with our nuisance parameters $$a$$ and $$s$$. First, we can __set them to their maximum a posteriori solutions__ $$a^{\mathrm{MAP}}$$ and $$s^{\mathrm{MAP}}$$, and compute $$L$$ with those values. This is equivalent to directly fitting for $$a$$ and $$s$$ at fixed $$\vec{x}$$, which is useful. (In this case, one needs to compute the covariance terms which I have omitted above).
Second, we can __marginalize__ over $$a$$ and $$s$$, since we have isolated there contributions and those are Gaussians! In other words, we can write
 
 $$\iint  p(\vec{y}|\vec{x}, \Sigma, s, a)p(s)p(a) \mathrm{d}s\mathrm{d}a =  \Bigl( (2\pi)^D \ F_\mathrm{TT}   \sigma^2_s \ |\Sigma_{ya}|  \Bigr)^{-1/2}  \exp\left( - \frac{1}{2} F_\mathrm{OO}   + \frac{1}{2} \frac{F_\mathrm{OT} ^2}{F_\mathrm{TT} } \right) $$
 
 This is very useful; as I said previously, those terms unavoidably appear in any Bayesian analysis, hierarchical or not, We can now focus on $$\vec{x}$$ and analytically marginalize over $$a$$ and $$s$$ when fitting $$\vec{y}$$, for example in each step of an MCMC algorithm constraining the parameters of $$\vec{x}$$. Sweet!
 

