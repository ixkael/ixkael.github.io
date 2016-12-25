---
id: 47
title: 'Installing S2LET, FLAG &#038; FLAGLET on Ubuntu/MacOSX'
date: 2013-01-01T23:18:41+00:00
author: admin
layout: post
guid: http://ixkael.com/blog/?p=47
permalink: /installing-s2let-flag-flaglet-on-ubuntumacosx/
categories:
  - Computing and Geekries
---
This tutorial is complementary to the generic one presented on [this page](http://ixkael.com/blog/) and will guide you through the installation of FFTW, CFITSIO, HEALPIX, SSHT, S2LET, FLAG and FLAGLET. The procedure was initially tested on a blank Ubuntu 12 virtual machine but will also work on Mac OSX systems (Lion minimum recommended). MATLAB >2011 is also recommended because the Mex compiler is painful to configure with old versions of gcc (see below).

<!--more-->

# Compiling and testing the C libraries

## FFTW

Download FFTW 3.x.x at the [official website](http://www.fftw.org/download.html). Unzip it and configure/install it in your favorite software directory (_/home/bl/software/fftw-3.3.3_ for me) with the following commands. The flag _–with-pic_ is required on Linux only.

```
    ./configure --prefix=/home/bl/software/fftw-3.3.3 --enable-shared --with-pic make make install
```

The should create _lib_ and _include_ folders in _/home/bl/software/fftw-3.3.3_ containing the FFTW libraries and headers.

## SSHT

SSHT can be obtained [here](http://www.mrao.cam.ac.uk/~jdm57/download.html). After unzipping the package, you should modify the makefile to use the right path to FFTW (and MATLAB, see below). In particular, you should pay attention to the following line and maybe add the explicit FFTW library:

```
    LDFLAGS = -L$(SSHTLIB) -l$(SSHTLIBNM) $(FFTWLIB)/lib$(FFTWLIBNM).a -lm
```

Then you should be able to run ‘make’ and test that SSHT works properly by running the test command:

```
    make
    ./bin/c/ssht_test 128
```

I am attaching the [makefile that worked for me]({{ site.baseurl }}/wp-content/uploads/2013/03/makefiles.zip), just in case.

## CFITSIO and Healpix

To use the IO and Healpix functionalties of S2LET, CFITSIO and HEALPIX_C are required. You can skip this paragraph if you don’t need them. Download and unpack CFITSIO from the [website](http://heasarc.gsfc.nasa.gov/fitsio/) and run

```
    ./configure make make install
```

If you don’t have it already (note that v2.x work well), download HEALPIX_3.00 from this [website](http://sourceforge.net/projects/healpix/) and run

```
    ./configure
```

Configure the C code only (i.e. main option 2) with the correct location for CFITSIO. In my case I had to specify _/home/bl/software/cfitsio/lib_ and _/home/bl/software/cfitsio/include_ during the configuration script. I didn’t apply any other change, option, or compiler flag since S2LET only need a few basic functions from the Healpix C library.

## S2LET / FLAG / FLAGLET

Note that theses three packages have identical setups (apart from their dependencies). S2LET/FLAG/FLAGLET can be obtained [here](http://www.mrao.cam.ac.uk/~jdm57/download.html). After unzipping the package(s), modify the makefile(s) with the right path pointing to FFTW and the other packages (CFITSIO, SSHT, etc). I had to change the order in the linking command and add fPIC to make it work on Linux. All the tests must run and achieve floating point accuracy.

```
    make ./bin/c/s2let_test
```

I am also attaching the [makefile that worked for me]({{ site.baseurl }}/wp-content/uploads/2013/03/makefiles.zip).

# Setting up the MATLAB interfaces

## GCC and the Mex compiler

A common problem is that the MEX compiler is only compatible with a few versions of gcc. For example I personally had gcc-4.7.2 installed and MEX was complaining a lot about it. So I had to

download an older version of GCC, compatible with MEX, to compile the MATLAB interfaces for SSHT/S2LET/FLAG/FLAGLET. For me, with MATLAB2012b the newest gcc compatible with mex was gcc-4.4. Also, MEX is annoying and not only needs this version installed, but also the right gcc and g++ symbolic links. The simplest solution was to rename /usr/bin/gcc into gcc-4.7.2 (my version) and make gcc-4.4 the default compiler by renaming it to /usr/bin/gcc. There are other ways to proceed and make Mex work with specific versions of GCC but it is far beyond the scope of this tutorial. Contact for further information. On Ubuntu I had to run something like:

```
    sudo apt-get install gcc-4.4
    sudo apt-get install g++-4.4
    sudo mv /usr/bin/gcc /usr/bin/gcc-4.7.2
    sudo cp /usr/bin/gcc-4.4 /usr/bin/gcc
    sudo cp /usr/bin/g++-4.4 /usr/bin/g++
```

Then MEX should find the right compilers and not complain anymore.

## SSHT/S2LET/FLAG/FLAGLET

If MATLAB is correctly pointed in the respective makefiles and the previous issue with the Mex compiler is resolved, then running the following command should be sufficient:

```
    make matlab
```

Most codes have demos (e.g. _s2let_demo1_, _flag_demo1_, etc) that you can run in MATLAB once you have added the correct directories to the PATH (e.g. $S2LET/src/main/matlab)
