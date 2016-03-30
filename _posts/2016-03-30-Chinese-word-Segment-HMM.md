---
layout: post
title:  "中文分词－HMM模型"
date:   2016-03-30 10:43:16 +0800
categories: Algorithm
#header-img: "img/post-bg-js-module.jpg"
tags:
    - Algorithm
    - 中文分词
    - HMM
---


### 中文分词－HMM模型

关于HMM模型的文章和资料有很多，各种公式推导看起来也是有点复杂。不过还是那句话，多看看～～～

#### 模型介绍
隐马尔可夫模型（Hidden Markov Model，HMM）是统计模型，它用来描述一个含有隐含未知参数的马尔可夫过程。其难点是从可观察的参数中确定该过程的隐含参数。然后利用这些参数来作进一步的分析，例如模式识别。

摘自百科，有点抽象的。一般书上的介绍主要有5个元素。

* StatusSet:隐含状态集
* ObserveSet:观察状态集
* InitProbMatrix:初始状态概率矩阵
* TranProbMatrix:隐含状态转移概率矩阵
* EmitProbMatrix:发射状态矩阵

HMM模型可以用来解决三种问题：

* 参数(StatusSet,TransProbMatrix,EmitRobMatrix,InitStatus)已知的情况下，求解观察值序列。(Forward-backward算法)
* 参数(ObservedSet,TransProbMatrix,EmitRobMatrix,InitStatus)已知的情况下，求解状态值序列。(viterbi算法)
* 参数(ObservedSet)已知的情况下，求解(TransProbMatrix,EmitRobMatrix,InitStatus)。(Baum-Welch算法)

第3种问题似乎很厉害的样子，知道一个参数可以求出3个。。[知乎上有个例子，关于抓赌场老千](http://zhihu.com/question/20962240/answer/33561657)，很形象。第2个问题，就是本文要介绍的，用来解决中文分词的问题。

关于HMM模型的一般原理介绍，可以看[52nlp](http://www.52nlp.cn/hmm%E7%9B%B8%E5%85%B3%E6%96%87%E7%AB%A0%E7%B4%A2%E5%BC%95)上的系列文章，很多，可以挑着看，再看后面的在中文分词应用。看不懂可以多看几遍。

#### 5大元素在中文分词中的具体含义。
**ObserveSet**
首先我们拿到一段文本：

> 小明硕士毕业于中国科学院计算所

这就是观察值序列，也就是我们看到的。。。好直白<code>P:</code>

**StatusSet**

在说说隐含状态集，顾名思义就是我们看不出来的；这里的隐含状态指的是每个字的状态。
那么每个字都有哪些状态。
有 词语的开头、词语的中间字、词尾、单个字，当然简单的分有 开头、结尾；
这里的隐含状态集有4个状态对应的英文字母
> {B,M,E,S}

显然这是我们要求解的。
输入

> 小明硕士毕业于中国科学院计算所

输出

> BEBEBMEBEBMEBES

根据这个来切分

> BE/BE/BME/BE/BME/BE/S

也就是
> 小明/硕士/毕业于/中国/科学院/计算/所

这里的4个状态之间的关系，不是两两任意组合的，比如B后面只能是M E，M 后面只能是 M E ，BS 这样是不存在的。

这就是输入，输出的过程，很简单，不过具体中间过程是什么样的，这里我们只是讲到了5个元素中的2个。下面再看看另外3个元素是怎么使用的。

#### InitStatusSet

初始状态概率分布

> B -0.57746333344078771
> 
> M -inf
> 
> E -inf
> 
> S -0.82398607534977875

也就是句子中第1个字是<code>{B,M,E,S}</code>的概率，显然句子开头只能是<code>B</code>，词语的开头或者<code>S</code>，单个字。
这里概率取了对数,也可以自己直接给个默认值，比如都是0.5，我们认为2种可能性一样；当然也可以自己统计下现有的文本，看看具体的概率是什么样的。

#### TransProbMatrix
隐含状态转移矩阵，也就是4个状态之间的转移概率，<code>BMES*BMES</code>,<code>4*4</code>的矩阵。

```
   B        M        E        S
B -inf    -1.8750  -0.1664  -inf
M -inf    -0.7227  -0.6644  -inf
E -0.6378 -inf     -inf     -0.7517
S -0.5051 -inf     -inf     -0.9248
```
这里的概率取了对数值，有一些负无穷的值是表示不可能的状态，比如M 后面不可能是B。

#### EmitProbMatrix
发射状态概率矩阵

```
   你        好        小        明   ...
B -0.6378  -1.8750  -6.2217   -inf
M -0.6378  -0.7227  -5.0638   -inf
E -0.6378 -0.6378   -8.2379   -0.7517
S -0.5051 -0.6378   -6.3409   -0.9248
```
这里的矩阵的横向是中文字，假如有5000个常用字，那么这个矩阵就是<code>4*5000</code>维。
通常这个矩阵很稀疏
，而且也不能保证所有的字都在这个矩阵里，所以一般是采用4个<code>map</code>来存储，如果<code>map</code>里不存在，那么给个很小的值，表示概率很低。

具体值的含义是<code>P(Observed[i]|Status[j])==P(你|B)</code>

ok,全部的5个元素介绍完了，那么理论上 假如我有了这些参数值([现成的参数模型](https://github.com/yanyiwu/cppjieba/blob/master/dict/hmm_model.utf8))，就可以拿来作分词了。分词属于前面提到的HMM用来解决的第2类问题，需要用Viterbi算法来求解。

### HMM－Viterbi算法
再次拿出待切分文本

> 小明硕士毕业于中国科学院计算所

对于这个15个字长度的文本，每个字有<code>{B,M,E,S}</code>4种状态。说白了，我们想要的就是，概率最大的一个<code>BMES...</code>组合，如果枚举一下，那么有<code>4^15</code>次方，显然我们不能让机器太累，毕竟地球变暖，人人有责~~<code>P:</code>.

那么我们可以这样，先确定第一个字

由**InitStatusSet**

初始状态概率分布

> B -0.57746333344078771
> 
> M -inf
> 
> E -inf
> 
> S -0.82398607534977875

和 **EmitProbMatrix**

> Status[B] -> Observed[小]: -6.2217
>
> Status[M] -> Observed[小]: -5.0638
> 
> Status[E] -> Observed[小]: -8.2379
> 
> Status[S] -> Observed[小]: -6.3409

weight[B,小]=－0.5774+ -6.2217=-6.7991

weight[M,小]=-inf+ -5.0638= -inf

weight[E,小]=-inf+ -8.2379= -inf

weight[S,小]=-0.8239+-6.3409= -7.1648


这里<code>小</code>有4种情况，已经计算好了；

这里<code>明</code>也有4种情况，那么如何计算呢？

<code>明</code>的隐状态概率和前面一个字相关，这也就是**HMM的重大假设:当前T=i时刻的状态Status(i)，只和T=i时刻之前的n个状态有关。**

这里简单处理就是，每个字符的状态只和前一个状态相关，构建一个2-gram(bigram)语言模型，也就是一个1阶HMM。当然这种假设是不切实际的，也是hmm的缺陷的地方，这里不作讨论。

那么<code>明</code>的状态依赖于<code>小</code>的状态，那么就有<code>4x4</code>种情况，取最大的概率。整个过程计算下来，大概是<code>4x4x15</code>次计算，比起枚举的<code>4^15</code>次
，大大减少计算量了。


关于**Viterbi**算法的具体公式推导可以看[52nlp](http://www.52nlp.cn/hmm%E7%9B%B8%E5%85%B3%E6%96%87%E7%AB%A0%E7%B4%A2%E5%BC%95)上相关的系列文章。

### Viterbi算法代码实现

变量解释

weight[4][15]表示，4种状态下的概率，15个字。

weight[0][1]表示<code>明</code>，在状态B下，出现的概率。

path[4][15]表示，4种状态下前一个字的状态，15个字。对于每个字我们都在求4种状态下的最大概率，那么相对应的就有前一个字的状态；
path[0][1]表示：在weight[0][1]取最大概率时，前面一个字的状态，比如path[0][1]=0，表示<code>小</code>的状态是B.
这个path用法好巧妙，通过记录前面一个字的状态，最后计算完整个weight数组，回溯回来，就得到整个结果。

``` c++

//维特比算法
void Viterbe(std::vector<uint32_t>& words, std::vector<size_t>& ret)
{
    std::vector< vector<double> > weight = std::vector< vector<double> >(4,vector<double>(words.size(),0));
    std::vector< vector<int> > path = std::vector< vector<int> >(4,vector<int>(words.size(),0));
    
    //计算第1个字的概率
    for (int i=0; i<4; i++) {
        weight[i][0] =Init_Status[i]+ GetEmitProb(*(vec_Emit_map[i]),words[0]);
    }
    
    for (int i=1; i<words.size(); i++) {
        for (int j=0; j<4; j++) {
            weight[j][i] = -INT64_MAX;
            path[j][i]=-1;
            for (int k=0; k<4; k++) {
                //前面一个字的状态概率＋状态转移概率＋发射矩阵概率
                //因为取对数 所以 ＋
                double tmp = weight[k][i-1]+TransProbMatrix[k][j]+GetEmitProb(*(vec_Emit_map[j]),words[i]);
                if (tmp>weight[j][i]) {
                    weight[j][i]=tmp;
                    path[j][i]=k;
                }
            }
        }
    }
    
    //计算完整个weight 最后一个字要么是 E 要么 S，比较一下 就可以确定。
    double endE,endS;
    endE = weight[2][words.size()-1];
    endS = weight[3][words.size()-1];
    int inx=-1;
    if (endE>endS)
        inx = 2;
    else
        inx = 3;
    ret.resize(words.size());
    
    //利用最后一个字的状态来回溯整个状态路径，很巧妙的运用了path数组
    for (int i=words.size()-1; i>=0; i--) {
        ret[i]=inx;
        inx = path[inx][i];
    }
}
```

### 模型训练

前面一直在假设 我们已经拥有了**InitStatusSet**，**TransProbMatrix**，**EmitProbMatrix** ，显然还是没法自己手写一个HMM分词器。网上有现成的模型([HMMModel](https://github.com/yanyiwu/cppjieba/blob/master/dict/hmm_model.utf8))

当然也可以自己写一个。首先下载一个已经标注好了的训练文本。
使用的语料来自于SIGHAN Bakeoff 2005的[icwb2-data.rar](http://www.sighan.org/bakeoff2005/data/icwb2-data.rar)
下载不了就百度吧～<code>P:</code>

> /icwb2-data.rar/training/msr_training.utf8  用以训练HMM，其中包含已分词汇约2000000个
> 
> /icwb2-data.rar/testing/pku_test.utf8           测试集

利用msr_training.utf8训练出**InitStatusSet**，**TransProbMatrix**，**EmitProbMatrix**；

然后利用**Viterbi**算法进行分词。
见代码实现[HMMSeg](https://github.com/y521263/HMMSeg)Demo.

#### 参考
[浅谈中文分词](http://www.isnowfy.com/introduction-to-chinese-segmentation/)

[HMM模型抓赌场老千](http://zhihu.com/question/20962240/answer/33561657)

[HMM相关文章索引](http://www.52nlp.cn/hmm%E7%9B%B8%E5%85%B3%E6%96%87%E7%AB%A0%E7%B4%A2%E5%BC%95)

[中文分词之HMM模型详解](http://www.yanyiwu.com/work/2014/04/07/hmm-segment-xiangjie.html)

[jieba分词(C++版本)](https://github.com/yanyiwu/cppjieba)

