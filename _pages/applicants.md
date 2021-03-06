---
layout: page
title: Information for prospective applicants
permalink: advice/applicants/
subtitle:
---

# Information for prospective applicants
<br/>
For the projects below, _all scientific backgrounds will be considered._

However, most (if not all) of the projects I propose involve some analysis and modeling of astronomical data, with a focus on either cosmology or Galactic astrophysics. Therefore, some experience in scientific programming and data analysis is very strongly recommended (for example in python, which is currently the language of choice in astrophysics). No specific experience is required in astronomy, cosmology, or theoretical physics, or even statistics or machine learning. But some experience or interest in at least some of those fields is strongly recommended.

Please also carefully read <a href="../expectations/"> this page</a> on what to expect when working with me.

## PhD students

Do not hesitate to reach out if you would like to discuss projects and opportunities. However, for more efficient discussions please first research the PhD admission process in the Astrophysics group, as well as possible funding sources (STFC, President's scholarship, etc). I am not describing specific projects here, but it will most likely be in observational cosmology, and involve analysis of real data with sophisticated statistical techniques theoretical modelling. For 2021 the list of available projects in the department <a href="https://www.imperial.ac.uk/astrophysics/students-and-prospective-students/phd-projects/">is available here</a>.

## MSc/MSci students

Every year I offer projects that last between 3 and 9 months, depending on the degree, and can be carried out alone or in pairs (but you will be responsible for splitting the work). _All degrees are welcome_. Typically, I expect the projects to attract data-minded students in the following MSc/MSci programs: Physics, Data Science, Computer Science, Artificial Intelligence, Electrical Engineering, etc. For the requirements, see the message on top of this page.

### List of possible projects (last update Jan 2021):
_Some of the projects below are complementary, so that we may have more vibrant and collaborative weekly group meetings, and students may help each other more, both directly and indirectly._
- **Systematics in galaxy clustering**: implement the systematics model of <a href="https://arxiv.org/pdf/2012.08467.pdf">this paper</a>, to learn a flexible relationship between the observed sky density of galaxies and nuisance observational systematics (Galactic foregrounds, atmospheric conditions, number of exposures, etc). Such density-systematics correlations are major obstacles in cosmological analyses and must be removed. This project will focus on implementing the simplest multivariate linear model with Self-Organised Maps (and potentially other unsupervised machine learning methods), possibly on the same data (KIDS DR4, or any other easily accessible public cosmology surveys) but not necessarily go all the way to final diagnostics with correlations functions and "random" synthetic catalogs (if anything, we would use angular power spectra instead).
- **Injecting simulated galaxies in photometric images**: review and play with one or multiple codes to perform galaxy injections and extractions (for example the public DECALS or LSST pipelines; <a href="https://arxiv.org/abs/2003.06090">this review paper</a> is a good start). Injection simulations are critical for accurately characterizing the image analysis pipelines that measure the properties of millions of galaxies (for testing models of cosmology and galaxy formation). This project would include a brief review of photometric image simulations: how synthetic galaxies are simulated (morphologies, colors, etc), how they are injected in photometric images (real or also synthetic), and how the galaxy properties of interest are measured/recovered by a real analysis pipeline in the presence of noise, artefacts, bright stars, other nearby galaxies, etc.
- **Measuring the acceleration of the solar system with Gaia EDR3 data**: repeat the analysis of <a href="https://arxiv.org/pdf/2012.02036.pdf">this paper</a> as closely as possible, and compare with a more conventional Bayesian version of the inference.
- **Quasars in Gaia EDR3 data**: reproduce the most recent Gaia quasar sample (EDR3 paper to come soon; in the meantime the <a href="https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..14G/abstract">DR2 paper<a> is a good start), investigate the level of stellar contamination, and construct a new catalog of quasar candidates that is not based on cross-matching (for example using a machine learning method or a simple density model, potentially following <a href="https://arxiv.org/abs/1910.05255"> this paper</a>).
- **Testing synthetic stellar population models on Lyman Break galaxy spectra**: applying the <a href="https://github.com/cconroy20/fsps">FSPS</a> and <a href="https://arxiv.org/abs/2012.01426">Prospector</a> codes to spectroscopy from a few dropout / Lyman Break galaxies (possibly from that <a href="https://arxiv.org/abs/2004.00158">data</a>) to extract physical properties of interest and compare with other methods.
- **Normalizing flows on quasars, stars, galaxies**: reproduce some of the results of <a href="https://arxiv.org/abs/2012.05220">this paper</a> (classifying galaxies, stars, and quasars from the Gaia data) using more sophisticated Bayesian machine learning techniques, such as normalising flows.
- **Comparison of Bayesian neural network methods**: review Bayesian neural network approaches (see this <a href="https://arxiv.org/pdf/2011.06225.pdf">recent review</a>), in particular deep ensembles and the recent work on <a href="https://arxiv.org/pdf/2007.05864.pdf">neural tangent kernels</a>. Apply to some simple astronomical problems of interest.
- **Solving astrophysics PDEs with machine learning**: use the <a href="https://arxiv.org/pdf/2010.08895.pdf">Fourier Neural Operator</a> (a revolutionary machine learning technique for solving PDEs) on simple applications in astrophysics and cosmology, to study its applicability.


## Postdocs

Our group can support your application for externally-funded fellowship with us. Examples include: Imperial JRF, Royal Society URF, STFC JRF, etc. Simply contact any of the staff members (including myself).
