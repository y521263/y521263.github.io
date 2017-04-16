---
layout: post
title:  "python tornado multiprocess multithreading"
date:   2017-04-16 10:43:16 +0800
categories: Python
#header-img: "img/post-bg-js-module.jpg"
tags:
    - Python 
    - tornado 
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## tornado

Tornado是Python生态中性能比较不错的Web服务框架。

Tornado 和现在的主流 Web 服务器框架（包括大多数 Python 的框架）有着明显的区别：它是非阻塞式服务器，而且速度相当快。得利于其 非阻塞的方式和对 epoll 的运用，Tornado 每秒可以处理数以千计的连接，这意味着对于实时 Web 服务来说，Tornado 是一个理想的 Web 框架。

最近需要用到tornado搭建一个web服务。其实本身性能已经很不错了，不过为了进一步提升性能，尝试了解了下，打开tornado多进程和多线程的姿势。

### multi-Process

``` python
import time
import tornado.ioloop
import tornado.web
import tornado.httpserver
from multiprocessing import cpu_count

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("this is main")

class MyHandler(tornado.web.RequestHandler):
    def get(self,n):
        #self.write("this is get")
        #time.sleep(n)
        for i in range(1000000000):
            a=i
        self.write("Awake! %s" % time.time())

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/sleep/(\d+)", MyHandler),
])
port=8888
if __name__ == "__main__":
	
    """
    #single process
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()
    """
    #multi process
    server = tornado.httpserver.HTTPServer(application)
    server.bind(port)
    #specify number of subprocess
    #server.start(4)
    server.start(cpu_count())
    tornado.ioloop.IOLoop.current().start()
```

首先可以试试单进程模式，main函数中上面的注释打开，下面的注释掉，分别在2个终端打开：

``` sh
curl "http://localhost:8888/sleep/5"
```

``` sh
curl "http://localhost:8888/"
```
就会发现第一次的curl很久都没有返回，而第二次的curl 也一直阻塞在那里。因为主进程在做一个超长的for循环(很耗CPU的work)

同样的，再试试multi-process模式，第二次的 curl很快就返回。

### multithreading

``` python
import time
import tornado.ioloop
import tornado.web
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps

EXECUTOR = ThreadPoolExecutor(max_workers=5)

def unblock(f):
	@tornado.web.asynchronous
	@wraps(f)
	def wrapper(*args, **kwargs):
		self = args[0]

		def callback(future):
			self.write(future.result())
			self.finish()

		EXECUTOR.submit(
			partial(f, *args, **kwargs)
		).add_done_callback(
			lambda future: tornado.ioloop.IOLoop.instance().add_callback(
				partial(callback, future)))
	return wrapper

class MainHandler(tornado.web.RequestHandler):
	def get(self):
		self.write("Hello, world %s" % time.time())

class SleepHandler(tornado.web.RequestHandler):
	@unblock
	def get(self, n):
		#time.sleep(float(n))
		for i in range(1000000000):
			a=i
		return "Awake! %s" % time.time()

application = tornado.web.Application([
	(r"/", MainHandler),
	(r"/sleep/(\d+)", SleepHandler),
])

if __name__ == "__main__":
	application.listen(8888)
	tornado.ioloop.IOLoop.instance().start()
```
多线程版本看起来比较复杂，这里解释下。

首先如果是python2.7环境，需要安装一下concurrent；

这里的unblock函数，作为一个装饰器(**decorator**)的概念出现，就是把所有的 get函数的操作交给线程池处理。执行完成后，有一个callback。在callback再把结果 <code>self.write</code>。
self.write并不是线程安全的，需要在主进程里操作。


这里只需要在，需要并行的get函数前，加上一行

``` python
	@unblock
	def get(self, n):
	...

```

那么这里，再次试验，发现，2次curl 都被阻塞了，过了很久都没返回。
是因为这里的for循环，很占CPU，即使多线程模式，依然被阻塞。如果这里把for循环，换成sleep(不占CPU的work)。那么就OK了。

可以根据实际业务需要，选择不同方式。

### 参考

1、[Blocking tasks in Tornado](https://lbolla.info/blog/2013/01/22/blocking-tornado)

2、[Python tornado web server: How to use multiprocessing to speed up web application](http://stackoverflow.com/questions/32273812/python-tornado-web-server-how-to-use-multiprocessing-to-speed-up-web-applicatio)

3、[tornado.cn](http://www.tornadoweb.cn/)

4、[tornado.org](http://www.tornadoweb.org/en/stable/)
