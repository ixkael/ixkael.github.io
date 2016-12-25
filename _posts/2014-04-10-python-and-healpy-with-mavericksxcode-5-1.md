---
id: 25
title: Python and Healpy with Mavericks/Xcode 5.1
date: 2014-04-10T23:07:21+00:00
author: admin
layout: post
guid: http://ixkael.com/blog/?p=25
permalink: /python-and-healpy-with-mavericksxcode-5-1/
categories:
  - Computing and Geekries
---
I am quite satisfied with my Python installation. I simply followed one of the numerous tutorials on the web and installed a virtual environment with Brew, easy_install and pip. However, I migrated to Mavericks recently, and started experiencing serious problems, especially with [Healpy](healpy.readthedocs.org). Strangely, I migrated three machines, which have the exact same Python installation, and only one had these problems. Anyways, the reason is that Apple Xcode 5.1 now throws errors when used with unknown compiler flags. The easy fix that worked for me is to defined the following symbols before running "pip install healpy" :


```    
    export CFLAGS=-Qunused-arguments
    export CPPFLAGS</span>=-Qunused-arguments
    export ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future
```

Sources: [[1]](http://kaspermunck.github.io/2014/03/fixing-clang-error/), [[2]](http://stackoverflow.com/questions/22413050/cant-install-python-mysql-library-on-mac-mavericks), [[3]](https://langui.sh/2014/03/10/wunused-command-line-argument-hard-error-in-future-is-a-harsh-mistress/).
