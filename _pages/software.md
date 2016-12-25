---
id: 9
title: 'Research &#038; Codes'
date: 2014-08-29T22:20:42+00:00
author: admin
layout: page
guid: http://ixkael.com/blog/?page_id=9
---
## Brief summary of my research<figure id="attachment_642" style="width: 429px" class="wp-caption aligncenter">

<img class="wp-image-642 size-large" src="http://ixkael.com/blog/wp-content/uploads/2014/08/13083165_1781933075361815_1978603390719965953_n-629x650.jpg" alt="13083165_1781933075361815_1978603390719965953_n" width="429" height="450" />Data-driven observational cosmology and the scientific method. 

[See my publications on arXiv](http://arxiv.org/find/all/1/all:+AND+boris+leistedt/0/1/0/all/0/1)! Some of my talks/presentations are available on [SpeakerDeck](https://speakerdeck.com/ixkael).

<div style="padding-top: 0px;">
</div>

My primary research interest is the the large-scale structure of the universe, which refers to the distribution and properties of matter traced by galaxies, observed during large surveys of the night sky. The large-scale structure contains a wealth of information about the content, the evolution and the origins of our universe, and can be used to address pressing questions such as the nature of dark matter and dark energy.

In addition to the large-scale structure, I also study the cosmic microwave background, the relic light of the Big Bang, and the astrophysics of quasars, distant galaxies devoured from the inside by supermassive black holes.

To exploit the data collected by modern telescopes, I develop robust techniques inspired from signal processing and information theory. In the era of precision cosmology, their purpose is to extract the most information from the data while accounting for all the uncertainties. My expertise include statistical methods and wavelet methods in two and three dimensions. Wavelets are very powerful tools for extracting localised features in both pixel and frequency space.

* * *

## Available software and resources

### Extensions to CLASS

<https://github.com/ixkael/class_public>

My fork of CLASS to support arbitrary galaxy samples: bias, magnitude bias, and redshift distributions stored as Gaussian mixtures of discrete histograms.

<img class="aligncenter size-large wp-image-447" src="http://ixkael.com/blog/wp-content/uploads/2014/08/class_nz_comp-650x200.png" alt="class_nz_comp" width="556" height="170" srcset="http://ixkael.com/blog/wp-content/uploads/2014/08/class_nz_comp-650x200.png 650w, http://ixkael.com/blog/wp-content/uploads/2014/08/class_nz_comp-300x92.png 300w, http://ixkael.com/blog/wp-content/uploads/2014/08/class_nz_comp-624x192.png 624w, http://ixkael.com/blog/wp-content/uploads/2014/08/class_nz_comp.png 1300w" sizes="(max-width: 556px) 100vw, 556px" />

* * *

### SDSS systematics templates

_Template maps for the potential image systematics in the SDSS DR8-10 data. Released as HEALPix maps at Nside=256. Please _

•   References : [[1404.6530]](http://arxiv.org/abs/1404.6530) and [[1405.4315]](http://arxiv.org/abs/1405.4315)
  
•   [Download (warning: 430 Mb file)](https://www.dropbox.com/s/a2u0n78m6p248vn/SDSS_DR10_systematics_templates.zip?dl=0)
  
•   [Announcement and info](http://ixkael.com/blog/release-of-the-sdss-systematics-templates/)

<img class="aligncenter wp-image-232" src="http://ixkael.com/blog/wp-content/uploads/2014/08/Field_PSFWIDTH_I_256_5-650x229.png" alt="Field_PSFWIDTH_I_256_5" width="556" height="196" srcset="http://ixkael.com/blog/wp-content/uploads/2014/08/Field_PSFWIDTH_I_256_5-650x229.png 650w, http://ixkael.com/blog/wp-content/uploads/2014/08/Field_PSFWIDTH_I_256_5-300x106.png 300w, http://ixkael.com/blog/wp-content/uploads/2014/08/Field_PSFWIDTH_I_256_5-624x220.png 624w, http://ixkael.com/blog/wp-content/uploads/2014/08/Field_PSFWIDTH_I_256_5.png 800w" sizes="(max-width: 556px) 100vw, 556px" />

* * *

### Wavelets

_The following code is publicly available to perform fast exact wavelet transforms on the sphere and on the ball. Visit <a href="http://www.mrao.cam.ac.uk/~jdm57/download.html" target="_blank">this page</a> to receive copies of the packages by email._

<div style="padding-top: 1px;">
  <ul>
    <li>
      <div style="padding-top: 5px;">
        <a href="http://www.s2let.org" target="_blank"><strong>S2LET</strong></a>: fast, exact scale-discretised wavelet transform. <a href="http://www.s2let.org/" target="_blank">www.s2let.org.</a>
      </div>
    </li>
    
    <li>
      <div style="padding-top: 5px;">
        <a href="http://www.mrao.cam.ac.uk/~jdm57/flag/index.html" target="_blank"><strong>FLAG</strong></a>: exact harmonic “Fourier-Laguerre” transform on the ball.
      </div>
    </li>
    
    <li>
      <div style="padding-top: 5px;">
        <a href="http://www.flaglets.org/" target="_blank"><strong>FLAGLET</strong></a>: exact wavelet transform on the ball. <a href="http://www.flaglets.org/" target="_blank">www.flaglets.org.</a>
      </div>
    </li>
  </ul>
  
  <p>
    Dependencies: <a href="http://www.fftw.org/" target="_blank"><strong>FFTW</strong></a> => <a href="http://www.mrao.cam.ac.uk/~jdm57/ssht/index.html" target="_blank"><strong>SSHT</strong></a> => {<a href="http://s2let.org/" target="_blank"><strong>S2LET</strong></a>, <a href="http://www.mrao.cam.ac.uk/~jdm57/flag/index.html" target="_blank"><strong>FLAG</strong></a>} => <a href="http://flaglets.org/" target="_blank"><strong>FLAGLET</strong></a>
  </p>
  
  <p>
    <img class="aligncenter wp-image-233" src="http://ixkael.com/blog/wp-content/uploads/2014/08/kernels-650x206.png" alt="kernels" width="533" height="169" srcset="http://ixkael.com/blog/wp-content/uploads/2014/08/kernels-650x206.png 650w, http://ixkael.com/blog/wp-content/uploads/2014/08/kernels-300x95.png 300w, http://ixkael.com/blog/wp-content/uploads/2014/08/kernels-624x198.png 624w, http://ixkael.com/blog/wp-content/uploads/2014/08/kernels.png 892w" sizes="(max-width: 533px) 100vw, 533px" />
  </p>
  
  <hr />
  
  <h3>
     3DEX
  </h3>
  
  <p>
    <strong>3D EXpansions</strong> is an open-source library for fast 3-dimensional Fourier-Bessel decomposition. The research paper was published in Astronomy & Astrophysics, and is available on astro-ph: <a href="http://arxiv.org/abs/1111.3591">http://arxiv.org/abs/1111.3591</a>. The library involves a novel formalism for the computation of Fourier-Bessel expansions, relying on the well-known <strong><a href="http://healpix.jpl.nasa.gov/">HEALPix</a> </strong>library, used in geophysics and astrophysics for 2D spherical harmonics decomposition. This library was developed in the context of an internship at the CEA Saclay in 2010-2011 under the supervision of <a href="http://web.me.com/anais.rassat/">Dr Anais Rassat</a> and <a href="http://jstarck.free.fr/jstarck/Home.html">Prof. Jean-Luc Starck</a>. Don’t hesitate to contact me in case of problem or for further information on 3DEX. I will be happy to help you and to add new suggested features.
  </p>
  
  <p>
    <strong>License</strong> : <a href="http://www.cecill.info/index.en.html">CeCILL</a><br /> <strong>Download</strong> : upon request<br /> <strong>Experimental</strong> : “3DEX” directory on <a title="3DEX on Github" href="https://github.com/ixkael/3DEX" target="_blank">my Github</a><br /> <strong>Requirements</strong> : HEALPix and LAPACK/BLAS. The later will be made optional soon.<br /> <strong>Installation</strong> : preferably using <a href="http://www.cmake.org/">CMake</a>, but you may also use <em>./configure</em> and <em>make</em> commands for a HEALPix-like installation (see README file)<br /> <strong>Documentation</strong> : see Documentation.html in the main directory<br /> <strong>Related research work</strong> : <a href="http://arxiv.org/abs/1111.3591">http://arxiv.org/abs/1111.3591</a>
  </p>
  
  <hr />
  
  <p>
    &nbsp;
  </p>
  
  <p>
  </p>
</div>