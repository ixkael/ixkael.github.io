---
id: 63
title: Installing SSHT, S2LET, FLAG and FLAGLET
date: 2012-12-25T23:58:24+00:00
author: admin
layout: post
guid: http://ixkael.com/blog/?p=63
permalink: /installing-ssht-s2let-flag-and-flaglet/
categories:
  - Computing and Geekries
tags:
  - wavelets
---
This is a post to guide you through the installation of the various packages required to perform exact wavelets on the sphere and on the ball. The first step is of course to <a href="http://www.mrao.cam.ac.uk/~jdm57/download.html" target="_blank">visit this page</a> to receive the previous packages by email. Once they are unzipped, you may follow the simple steps below.

<!--more-->

## Step 1 : Installing FFTW and optionally CFITSIO and HEALPIX

SSHT, thus all the other packages, require FFTW. The later can be downloaded <a href="http://www.fftw.org/download.html" target="_blank">here</a> and the installation should be straightforward using the usual _./configure_ and _sudo make all install_ commands.

If you intend to use the FITS and HEALPix features of S2LET, you should also download and install <a href="http://heasarc.gsfc.nasa.gov/fitsio/" target="_blank">CFITSIO</a> and <a href="http://healpix.jpl.nasa.gov/" target="_blank">HEALPIX</a>. The Fortran HEALPix library must be compiled with consistent Fortran flags; you will have to use the same flags to compile the Healpix Fortran interface in S2LET.

## Step 2 : Creating symbols in Bash or Shell

One only need to check a few lines in each Makefile to be able to build all packages. In brief, the things that potentially need to be adapted are the locations of the dependencies and the compilers with their options. But the very first step before looking at the makefiles is to define symbols pointing to the libraries. For example in Bash you can modify the following pattern and copy it to your _~/.profile_ or _~/.bashrc_ :


```    
    export FFTW='~/sofware/fftw'
    export SSHT='~/sofware/ssht'
    export S2LET='~/sofware/s2let'
    export FLAG='~/sofware/flag'
    export FLAGLET='~/sofware/flaglet'
```

If you intend to use the FITS and HEALPix features of S2LET, you should also define

```
    export CFITSIO='~/sofware/cfitsio'
    export HEALPIX='~/sofware/healpix'
```

and make sure the libraries are located in _/lib_ subdirectories.

## Step 3 : Setting up the makefiles

In the makefile for SSHT you must specify the location of FFTW, the C compilers+options and optionally the Matlab Mex compiler if you intend to use the Matlab interfaces. More details can be found below. On the contrary, in the makefiles for S2LET/FLAG/FLAGLET you should only check the compilers+options, since the makefiles will find the dependencies based on symbols defined in Bash/Shell.

All packages have Matlab interfaces that use Mex/C functions and must be built using the Matlab Mex compiler. Hopefully only the location of Matlab should be specified. For example on my machine it is

```
    # Directory for MATLAB
    MLAB = /Applications/MATLAB_R2011b.app
```

If you work on a recent Mac you should be able to build all libraries by running _make all_. On a Linux you may have to change a few options. In case you need to make these changes, below are the standard settings that work on Mac OS.

The standard settings for compiling the C libraries are

```
    # Compilers and options for C
    CC = gcc
    OPT = -Wall -O3 -g
```

If you want to use HEALPix in S2LET, the following flags must agree with those used to build HEALPix:

```
    # Compilers and options for Fortran FCC = gfortran OPTF90 = -O3 -ffree-form
    # To be defined if LGFORTRAN cannot be found in the path
    # GFORTRANLIB = /sw/lib/gcc4.6/lib
    # Flags for Healpix
    HPXOPT = -lgfortran -DGFORTRAN -fno-second-underscore -fopenmp
```

The following options will be used to compile the dynamic library required for the IDL interface in S2LET:

```
    # Config for dynamic library
    ifeq ($(UNAME), Linux) DYLIBEXT = so
    DYLIBCMD = cc -flat_namespace -undefined suppress endif
    ifeq ($(UNAME), Darwin) DYLIBEXT = dylib
    DYLIBCMD = g++ -flat_namespace -dynamiclib -undefined suppress endif
```

## Step 4 : Installing the libraries and interfaces

The packages must be built in this order: SSHT, S2LET, FLAG, FLAGLET. With valid dependencies and options, the command

111

should build the C library in every package. If the compilation was successful, you should check the exactness of each transform, for example by running

```
    $S2LET/bin/s2let_test
    $FLAG/bin/flag_test
    $FLAGLET/bin/flaglet_test
```

The command

```
    make all
```

will build the C library, the Matlab interfaces and the various high-level programs provided in every package.
