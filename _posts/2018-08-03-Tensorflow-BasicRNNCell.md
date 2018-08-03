---
layout: post
title:  "Tensorflow BasicRNNCell"
date:   2018-08-03 09:43:16 +0800
categories: RNN
#header-img: "img/post-bg-js-module.jpg"
tags:
    - RNN 
    - Tensorflow 
    - LSTM 

---


## RNN

关于RNN的文章，colah的blog这篇文章讲得很详细。从基本的RNN到LSTM以及各种变种。

![](/img/article/2018-08-03-RNN-unrolled.png)

不过关于RNN里num_units的含义，weight参数的数量计算，没具体提到。

首先来看下Tensorflow里[RNN](https://github.com/tensorflow/tensorflow/blob/be3d22844025e42e177a21479f3ae73bc5351c1f/tensorflow/python/ops/rnn_cell_impl.py#L333)的实现。


``` python

@tf_export("nn.rnn_cell.BasicRNNCell")
class BasicRNNCell(LayerRNNCell):
  """The most basic RNN cell.
  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               name=None,
               dtype=None):
    super(BasicRNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output

```

这里weight大小为(input_depth+ \_num\_units)*\_num\_units

```shape=[input_depth + self._num_units, self._num_units]```

假设X的大小为[T,10],每个时序t对应的Xt：[1,10], input_depth=10

input_depth:也就是上图中Xt的输入大小。那么\_num\_units上图中没有体现出来，再看下图。

![](/img/article/2018-08-03-num_units.png)

这里相当于把上图展开看，里面藏着\_num\_units个隐藏单元。
那么相应的weight数量：\_num\_units* input_depth，再加上上一次t-1输出的hidden state。大小为\_num\_units;

weight=(input_depth+ \_num\_units)*\_num\_units

bias=\_num\_units

类似的[LSTM](https://github.com/tensorflow/tensorflow/blob/be3d22844025e42e177a21479f3ae73bc5351c1f/tensorflow/python/ops/rnn_cell_impl.py#L519)

``` python
@tf_export("nn.rnn_cell.BasicLSTMCell")
class BasicLSTMCell(LayerRNNCell):
...
  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True
...

```
LSTM的参数数量，RNN的基础上再乘以4，也就是3个门以及tanh layer。

![](/img/article/2018-08-03-LSTM3-chain.png)


### 参考

1、[https://github.com/tensorflow/](https://github.com/tensorflow/)

2、[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

3、[https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell](https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell)

4、[https://www.knowledgemapper.com/knowmap/knowbook/jasdeepchhabra94@gmail.comUnderstandingLSTMinTensorflow(MNISTdataset)](https://www.knowledgemapper.com/knowmap/knowbook/jasdeepchhabra94@gmail.comUnderstandingLSTMinTensorflow(MNISTdataset))
