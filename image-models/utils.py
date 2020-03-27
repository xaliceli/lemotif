"""
utils.py
IO utilities and custom operations.
"""

import numpy as np
import scipy.stats as st
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras import models, initializers, regularizers, constraints


def gauss_kernel(blur_kernel, nsig=3):
    interval = (2*nsig+1.)/(blur_kernel)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., blur_kernel+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    kernel = tf.expand_dims(tf.stack([kernel, kernel, kernel], axis=2), axis=3)
    return tf.cast(kernel, tf.float32)


def minibatch_stddev_layer(x, batch_size):
    _, h, w, _ = x.get_shape().as_list()
    new_feat_shape = [batch_size, h, w, 0]
    mean, var = tf.nn.moments(x, axes=[0], keep_dims=True)
    stddev = tf.sqrt(tf.reduce_mean(var, keepdims=True))
    new_feat = tf.tile(stddev, multiples=new_feat_shape)
    return tf.concat([x, new_feat], axis=3)


def add_noise(x, batch_size):
    """
    Adds noise to outputs of convolutional layer.
    Args:
        x (tensor): Feature map.
    Returns:
        Feature map with stochastic variation introduced.
    """
    # Add randomly generated single-channel noise images of the same shape as current layer.
    shape = x.get_shape().as_list()
    noise = tf.random.normal([batch_size, shape[1], shape[2], 1], dtype=x.dtype)
    # Apply per-channel scaling factor to noise input through element-wise multiplication.
    noise_scaled = kl.Conv2D(filters=shape[-1], kernel_size=1, padding='same',
                             kernel_initializer=initializers.Zeros())(noise)
    x = kl.Add()([x, noise_scaled])

    return x


def instance_norm(x, epsilon=1e-8):
    """
    Instance normalization of x.
    Args:
        x (tensor): Layer to normalize.
        epsilon (float): Parameter adjusting the standard deviation of x.
    Returns:
        Normalized layer.
    """
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)
    x = tf.cast(x, orig_dtype)

    return x


class ELRDense(kl.Layer):
    """
    Custom layer for fully connected layer using learning rate equalization.
    """

    def __init__(self, units, gain=np.sqrt(2), lrmul=1.0, bias=True, name=None, **kwargs):
        self.filters = units if isinstance(units, int) else units.value
        self.gain = gain
        self.lrmul = lrmul
        self.use_bias = bias
        self.he_std = None
        self.bias = None
        super(ELRDense, self).__init__(name=name)

    def build(self, input_shape):
        fan_in = np.prod(input_shape.as_list()[1:])
        self.he_std = self.gain / np.sqrt(fan_in)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(fan_in, self.filters),
                                      initializer=initializers.RandomNormal(0, 1.0 / self.lrmul),
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters),
                                        initializer=initializers.Zeros(),
                                        trainable=True)
        super(ELRDense, self).build(input_shape)

    def call(self, x):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        out = tf.matmul(x, self.kernel * self.he_std * self.lrmul)
        if self.use_bias:
            if len(x.get_shape().as_list()) > 2:
                self.bias = tf.reshape(self.bias, [1, 1, 1, -1])
            out = out + self.bias * self.lrmul
        return out

    def get_config(self):
        config = {'filters': self.filters,
                  'gain': self.gain,
                  'lrmul': self.lrmul,
                  'bias': self.use_bias}
        base_config = super(ELRDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.filters)


class ELRConv2D(kl.Layer):
    """
    Custom layer for 2D convolution layer using learning rate equalization.
    """

    def __init__(self, kernel_size, filters, padding, gain=np.sqrt(2), lrmul=1.0, bias=True, name=None, **kwargs):
        self.kernel_size = kernel_size
        self.filters = filters if isinstance(filters, int) else filters.value
        self.padding = padding
        self.gain = gain
        self.lrmul = lrmul
        self.use_bias = bias
        self.he_std = None
        self.bias = None
        super(ELRConv2D, self).__init__(name=name)

    def build(self, input_shape):
        fan_in = self.kernel_size * self.kernel_size * input_shape.as_list()[-1]
        self.he_std = self.gain / np.sqrt(fan_in)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(
                                          self.kernel_size, self.kernel_size, input_shape.as_list()[-1], self.filters),
                                      initializer=initializers.RandomNormal(0, 1.0 / self.lrmul),
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters),
                                        initializer=initializers.Zeros(),
                                        trainable=True)
        super(ELRConv2D, self).build(input_shape)

    def call(self, x):
        out = tf.nn.conv2d(x, self.kernel * self.he_std * self.lrmul, strides=[1, 1, 1, 1],
                           padding=self.padding.upper())
        if self.use_bias:
            if len(x.get_shape().as_list()) > 2:
                self.bias = tf.reshape(self.bias, [1, 1, 1, -1])
            out = out + self.bias * self.lrmul
        return out

    def get_config(self):
        config = {'kernel_size': self.kernel_size,
                  'filters': self.filters,
                  'padding': self.padding,
                  'gain': self.gain,
                  'lrmul': self.lrmul,
                  'bias': self.use_bias}
        base_config = super(ELRConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

