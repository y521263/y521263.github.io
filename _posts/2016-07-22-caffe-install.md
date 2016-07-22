---
layout: post
title:  "caffe安装-Mac"
date:   2016-07-22 10:43:16 +0800
categories: Caffe
#header-img: "img/post-bg-js-module.jpg"
tags:
    - Caffe
---


## caffe安装 

Mac 10.11.2

###更新brew

```
brew update 
brew upgrade
```

###安装依赖

```
brew install -vd snappy leveldb gflags glog szip lmdb
# need the homebrew science source for OpenCV and hdf5
brew tap homebrew/science
brew install hdf5 opencv
```

###修改配置

```
brew edit opencv
```

将下面2行

```
args << "-DPYTHON#{py_ver}_LIBRARY=#{py_lib}/libpython2.7.#{dylib}"
args << "-DPYTHON#{py_ver}_INCLUDE_DIR=#{py_prefix}/include/python2.7"
```

替换为

```
args << "-DPYTHON_LIBRARY=#{py_prefix}/lib/libpython2.7.dylib"
args << "-DPYTHON_INCLUDE_DIR=#{py_prefix}/include/python2.7"
```

这里任选一个吧，我选了第2步

```
# with Python pycaffe needs dependencies built from source
brew install --build-from-source --with-python -vd protobuf
brew install --build-from-source -vd boost boost-python
# without Python the usual installation suffices
brew install protobuf boost
```

###编译

```
cp Makefile.config.example Makefile.config
```
修改<code>Makefile.config</code>里的
```
取消注释
CPU_ONLY :＝ 1
```
因为我这里只有cpu，没有显卡.cuda都没有装。。。

```
make all
make test
make runtest
```
ok 以上make没有报错就好了。
ps:官方文档仅供参考。有点古老了。

###参考

[http://caffe.berkeleyvision.org/](http://caffe.berkeleyvision.org/)

[How to install Caffe on Mac OS X 10.10 for dummies (like me)](http://hoondy.com/2015/04/03/how-to-install-caffe-on-mac-os-x-10-10-for-dummies-like-me/)

[MAC OS X10.10下Caffe无脑安装（CPU ONLY)](http://blog.csdn.net/yjn03151111/article/details/46353013)