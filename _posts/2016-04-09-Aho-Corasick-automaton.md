---
layout: post
title:  "AC自动机"
date:   2016-04-09 10:43:16 +0800
categories: Algorithm
#header-img: "img/post-bg-js-module.jpg"
tags:
    - Algorithm
---


### AC自动机

**Aho-Corasick automaton**,该算法在1975年产生于贝尔实验室，是著名的多模匹配算法之一。比如，给出一组单词，再给出一组字符串，找出有多少个单词在这个字符串中出现了。当然最简单的方法是逐个去找～<code>:P</code>


#### 预备知识
* [KMP](http://maples.me/algorithm/2016/02/16/KMP/)
* Trie

AC自动机需要以KMP、Trie字典树为基础来实现。不了解的话可以先看下这个2个。

KMP算法是单模式的字符匹配算法，AC自动机是多模式串的字符匹配算法，可以理解为是KMP加强版😄。

#### AC自动机的实现过程

我们给出5个单词，<code>say</code>,<code>she</code>,<code>shr</code>,<code>he</code>,<code>her</code>.给出字符串yasherhs.要找出有多少个单词在这个字符串中出现。

* 用这5个单词，构造Trie字典树
* 构造失败指针
* 扫描字符串进行匹配

#### Trie

首先我们需要建立一棵<code>Trie</code>。这棵树和一般的<code>Trie</code>不一样，多一个<code>fail</code>指针。

那么这里有2个指针。

* p 指向当前匹配的字符。
* p->fail p的失败指针，当p->next[i]为空的时候，可以跳转到p->fail继续匹配,如果没有，则指向root

对于Trie树的一个节点，对应一个序列s[1...m]。即p指向s[m]，那么接下来的下一个字符s[m+1]有2种情况。

1.p->next[s[m+1]] 不为空，那么就继续匹配。

2.p->next[s[m+1]] 如果为空，则跳转到失败指针<code>p=p->fail</code> 然后再重复1或2。如果失败指针为空，那么就匹配结束。

这里有个<code>p</code>指针转移的过程，**如果是一般的<code>Trie</code>树匹配，那么在2的情况下<code>p－>next[s[m+1]]==NULL</code>,匹配直接结束了，就说找不到单词s[1...m+1]。**

那么这里p指针转移的目的是：
**虽然单词s[1...m+1]不存在，也许单词s[i...m+1]存在，所以我们可以让p跳到某个节点，并且这个节点对应的字符串是s[i...m]，然后以s[i...m]为前缀继续匹配s[i...m+1]**。这里的s[i...m]是s[1...m]的后缀，i具体是多少这里暂时先不管。

这里用失败指针跳转就有<code>KMP</code>的next数组的味道了。

这里讲了失败指针的作用。下面再说如何构造。

#### 构造失败指针

首先从root开始，root的失败指针设置为空，因为本身这个节点也没有实际意义。

还要有一个队列<code>q</code>，存储待构造失败指针的节点

那么对于root的子节点<code>s</code>,<code>h</code>,第一个字符匹配失败也不需要跳转，因此将失败指针指向root，对应图中虚线(1)(2)。构造完<code>s</code>,<code>h</code>的失败指针，还需要把它们的子节点放入队列，也就是e,a,h.

然后第2次循环，队列弹出e节点，接下来p指向h节点的fail指针指向的节点，也就是root;此时<code>p==root</code>，<code>p->next['e'] == NULL</code>,所以<code>p=p->fail</code>,也就是NULL，因为<code>root->fail == NULL</code>.那么就把e的失败指针指向root,对应图中虚线(3)。再看看另外一个h(图中左边这个)，p指向s节点的fail指针，也就是root;此时<code>p==root</code>，<code>p->next['h'] != NULL</code>,有点不一样了，那么就把左边的h节点的失败指向右边的h节点，对应图中虚线(5).以此类推做完e,a h的失败指针构造，然后要把对应的子节点放入q队列。再做下一次循环，直到q队列为空，失败指针构造结束。

[盗个图~😄](http://blog.csdn.net/niushuai666/article/details/7002823)

![Trie_Fail](https://raw.githubusercontent.com/y521263/y521263.github.io/master/img/article/ac_Trie_Fail.png)

``` c++
void TrieTree::setFail()
{
    //使用队列 利用bfs遍历完所有节点
    std::queue<TrieNode*> q;
    q.push(root);
    while (!q.empty()) {
        TrieNode* tmp = q.front();
        TrieNode* p = tmp->fail;
        q.pop();
        for (int i=0; i<CHILDNUM; i++) {
            if( !tmp->next[i] ) continue;
            //父节点的失败指针 作为子节点的失败指针
            p = tmp->fail;
            
            //当p->next[i]为空 继续寻找fail指向的节点
            while (p && !p->next[i]) p = p->fail;
            
            tmp->next[i]->fail = p ? p->next[i]:root;
            
            //子节点放入队列 待后续构造
            q.push(tmp->next[i]);
        }
    }
}

```

#### 匹配

匹配过程分2种情况

1.当前字符匹配，p指针移到下一个字符继续匹配
2.当前字符不匹配，p指针移到失败指针所指向的字符继续匹配，直到指针指向root结束。

重复这2个过程，直到模式串结束。

拿出匹配串<code>yasherhs</code> i=0,1时，Trie没有对应的路径，然后继续往后移i=2,3,4.指针p走到左边的e节点。然后遍历失败节点，从左边的e跳到右边的e，然后再跳到root。整个过程发现有2个单词，就是红色圈圈的，跳转过程中发现有单词，可以记录下来。
然后i=5，此时p指向的还是刚才左下角的e，那么<code>p->next['r']==NULL</code>，那么<code>p=p->fail</code>,p指导右边的e，此时<code>p->next['r']!=NULL</code>,<code>p=p->next['r']</code>,然后就是和刚才一样，遍历失败指针。

``` c++
void TrieTree::query(std::string& s,std::set<std::string>& ret)
{
    TrieNode* p = root;
    for (int i=0; i<s.size(); i++) {
        //匹配不上 则使用失败指针
        while( p && !(*p)[s[i]]) p = p->fail;
        
        //这里的 p不动 下次循环再次从这个节点开始匹配
        p = p ? (*p)[s[i]] : root;
        
        //遍历完所有的失败节点，并且记录 所有找到的字符串
        for (TrieNode* tmp = p; tmp; tmp = tmp->fail) {
            if (tmp->isWord) {
                ret.insert(tmp->word);
            }
        }
    }
}

```

[代码](https://github.com/y521263/AlgorithmSet)

#### 参考

[AC自动机总结](http://blog.csdn.net/mobius_strip/article/details/22549517)

[AC自动机小结](http://www.cnblogs.com/kuangbin/p/3164106.html)

[AC自动机算法](http://blog.csdn.net/niushuai666/article/details/7002823)