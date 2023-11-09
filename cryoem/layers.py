
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
#from tensorflow_similarity.types import FloatTensor, IntTensor
from typing import Any, Dict, Optional
import tensorflow_addons as tfa
import math
from scipy import fftpack
import numpy.fft as fp

class BlurLayer(tf.keras.layers.Layer):
    def __init__(self, step, size, **kwargs):
        super(BlurLayer, self).__init__(**kwargs)
        self.step = step
        self.size = size

    def get_config(self):
        config = super(BlurLayer, self).get_config().copy()
        config.update({
            'step': self.step,
            'size': self.size
        })
        return config

    def build(self, input_shape):
        # Keeping track of the input shape to ensure output shape consistency
        self.input_shape_ = input_shape
        # Creating the dd variable here
        self.dd = self.add_weight(name="dd",
                                  shape=(input_shape[1], input_shape[2]),
                                  dtype=tf.int32,
                                  initializer="zeros",
                                  trainable=False)

    def low_pass(self, inputs):
        ims = []
        (n, w, h, c) = self.input_shape_
        im = tf.squeeze(inputs)
        im = tf.cast(im, dtype=tf.complex64)
        im = tf.reshape(im, [-1, w, h])
        half_w, half_h = int(w / 2), int(h / 2)
        us = 9 + half_w - np.geomspace(10, half_w - 1, self.step)
        for u in us:
            u = tf.cast(u, tf.int32)
            self.dd.assign(tf.zeros([w, h], tf.int32))
            freq = tf.signal.fft2d(im)
            freq1 = tf.identity(freq)
            freq2 = tf.signal.fftshift(freq1)
            freq2_low = tf.identity(freq2)
            block = tf.cast(self.dd[half_w - u:half_w + u + 1, half_h - u:half_h + u + 1].assign(tf.ones([2 * u + 1, 2 * u + 1], tf.int32)), tf.bool)
            freq2_low = tf.where(block, tf.cast(0, dtype=tf.complex64), freq2_low)
            freq2 -= freq2_low
            im1 = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(freq2)))
            ims.append(im1)
        ims.append(tf.squeeze(inputs))
        # Ensure that the output has the same number of channels as the input
        return tf.stack(ims, axis=-1)

    def call(self, inputs, **kwargs):
        outputs = self.low_pass(inputs)
        return outputs


class BlurLayer_exp(tf.keras.layers.Layer):
    def __init__(self, start, step, size, **kwargs):
        super().__init__()
        super().__init__(**kwargs)
        self.step = step
        self.size = size
        self.dd = tf.Variable(lambda: tf.zeros([self.size,self.size], tf.int32), trainable=False)
        self.s = tf.Variable(initial_value=lambda: tf.ones([1], tf.float32)*start, constraint=lambda x: tf.clip_by_value(x, 1.0, int(self.size/2)-2.0), trainable=True, name="var_s") #512
        self.stop = int(self.size/2)-1.0
        self.start = start

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'start': self.s.numpy(),
            'step': self.step,
            'size': self.size
        })
        return config

    def build(self, input_shape):
        pass

    def low_pass(self, inputs):
        ims = []
        (n, w, h, c) = inputs.shape
        #print(n,w,h,c)
        im = tf.cast(tf.convert_to_tensor(value=tf.squeeze(inputs)), dtype=tf.complex64)
        im = tf.reshape(im, [-1, w, h])
        half_w, half_h = int(w/2), int(h/2)
        ims = inputs
        for u in tf.linspace(self.s, self.stop, self.step): #validation error
          self.dd.assign(tf.zeros([w,h], tf.int32))
          tf.autograph.experimental.set_loop_options(shape_invariants=[(ims, tf.TensorShape([None, self.size, self.size, None]))])
          u = int(tf.squeeze(u))
          freq = tf.signal.fft2d(im)
          freq1 = tf.identity(freq)
          freq2 = tf.signal.fftshift(freq1)
          freq2_low = tf.identity(freq2)
          block = tf.cast(self.dd[half_w-u:half_w+u+1,half_h-u:half_h+u+1].assign(tf.ones([2*u+1,2*u+1], tf.int32)), tf.bool)
          freq2_low = tf.where(block, tf.cast(0, dtype=tf.complex64), freq2_low) # block the lowfrequencies
          freq2 -= freq2_low # select only the first 20x20 (low) frequencies, block the high frequencies
          im1 = tf.math.real(tf.signal.ifft2d(tf.signal.ifftshift(freq2)))
          #ims.append(im1)
          ims = tf.concat([ims, im1[:,:,:,tf.newaxis]], -1)
        #ims.append(tf.squeeze(inputs))
        self.add_metric(self.s, "s")
        return tf.reshape(ims, [-1, w, h, self.step+1])
        #return tf.stack(ims, axis=-1)

    def call(self, inputs, **kwargs):

        outputs = self.low_pass(inputs)

        return outputs


class BlurLayer2(tf.keras.layers.Layer):
    def __init__(self, step, size, **kwargs):
        super().__init__(**kwargs)
        self.step = step
        self.size = size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'step': self.step,
            'size': self.size
        })
        return config

    def build(self, input_shape):
        self.input_shape_ = input_shape

    def low_pass(self, inputs):
        lst = list(np.geomspace(0.1, 10, self.step))
        ims = []
        ims.append(tf.squeeze(inputs))
        (n, w, h, c) = self.input_shape_
        inputs = tf.reshape(inputs, [-1, w, h, c])
        for u in lst:
            im1 = tfa.image.gaussian_filter2d(inputs, [self.size, self.size], float(u))
            ims.append(tf.reshape(im1, [-1, w, h]))
        return tf.stack(ims, axis=-1)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, tf.Tensor):
            raise ValueError("Input should be a Tensor")
        outputs = self.low_pass(inputs)
        return outputs


# Encorporate denoise image
class BlurLayer3(tf.keras.layers.Layer):
    def __init__(self, step, size, **kwargs):
        super().__init__(**kwargs)
        self.step = step
        self.size = size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'step': self.step,
            'size': self.size
        })
        return config

    def build(self, input_shape):
        self.input_shape_ = input_shape

    def low_pass(self, inputs):
        lst = list(np.geomspace(0.1, 10, self.step))

        ims = []
        (n, w, h, c) = self.input_shape_
        inputs = tf.reshape(inputs, [-1, w, h, c])
        input0, input1 = tf.split(inputs, [1, 1], -1)
        ims.append(tf.squeeze(input0))
        ims.append(tf.squeeze(input1))
        for u in lst:
            im1 = tfa.image.gaussian_filter2d(input0, [self.size, self.size], float(u))
            ims.append(tf.reshape(im1, [-1, w, h]))
        return tf.stack(ims, axis=-1)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, tf.Tensor):
            raise ValueError("Input should be a Tensor")
        outputs = self.low_pass(inputs)
        return outputs

class GeneralizedMeanPooling(tf.keras.layers.Layer):
    def __init__(self, p: float = 3.0, data_format: Optional[str] = None, keepdims: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p = tf.Variable(initial_value=lambda: tf.ones([512], tf.float32)*p, trainable=True, name="var_p") #512
        #self.p = p
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.keepdims = keepdims

        #if tf.math.abs(self.p) < 0.00001:
        #    self.compute_mean = self._geometric_mean
        #elif self.p == math.inf:
        #    self.compute_mean = self._pos_inf
        #elif self.p == -math.inf:
        #    self.compute_mean = self._neg_inf
        #else:
        self.compute_mean = self._generalized_mean

    def compute_output_shape(self, input_shape):
        output_shape: IntTensor = self.gap.compute_output_shape(input_shape)
        return output_shape

    def call(self, inputs):
        raise NotImplementedError

    def _geometric_mean(self, x):
        x = tf.math.log(x)
        x = self.gap(x)
        return tf.math.exp(x)

    def _generalized_mean(self, x):
        e = self.p[tf.newaxis, tf.newaxis, tf.newaxis, :]
        x = tf.math.pow(x, e)
        x = self.gap(x) #512
        return tf.math.pow(x, 1.0 / self.p)

    def _pos_inf(self, x):
        raise NotImplementedError

    def _neg_inf(self, x):
        return self._pos_inf(x * -1) * -1

    def get_config(self) -> Dict[str, Any]:
        config = {
            "p": self.p.numpy(),
            "data_format": self.data_format,
            "keepdims": self.keepdims,
        }
        base_config = super().get_config()
        return {**base_config, **config}

class GeneralizedMeanPooling2D2(GeneralizedMeanPooling):
    r"""Computes the Generalized Mean of each channel in a tensor.
    $$
    \textbf{e} = \left[\left(\frac{1}{|\Omega|}\sum_{u\in{\Omega}}x^{p}_{cu}\right)^{\frac{1}{p}}\right]_{c=1,\cdots,C}
    $$
    The Generalized Mean (GeM) provides a parameter `p` that sets an exponent
    enabling the pooling to increase or decrease the contrast between salient
    features in the feature map.
    The pooling is equal to GlobalAveragePooling2D when `p` is 1.0 and equal
    to MaxPool2D when `p` is `inf`.
    This implementation shifts the feature map values such that the minimum
    value is equal to 1.0, then computes the mean pooling, and finally shifts
    the values back. This ensures that all values are positive as the
    generalized mean is only valid over positive real values.
    Args:
      p: Set the power of the mean. A value of 1.0 is equivalent to the
        arithmetic mean, while a value of `inf` is equivalent to MaxPool2D.
        Note, math.inf, -math.inf, and 0.0 are all supported, as well as most
        positive and negative values of `p`. However, large positive values for
        `p` may lead to overflow. In practice, math.inf should be used for any
        `p` larger than > 25.
      data_format: One of `channels_last` (default) or `channels_first`. The
        ordering of the dimensions in the inputs.  `channels_last`
        corresponds to inputs with shape `(batch, steps, features)` while
        `channels_first` corresponds to inputs with shape
        `(batch, features, steps)`.
      keepdims: A boolean, whether to keep the temporal dimension or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.  If `keepdims` is `True`, the temporal
        dimension are retained with length 1.  The behavior is the same as
        for `tf.reduce_max` or `np.max`.
    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`
    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, features)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          3D tensor with shape `(batch_size, 1, features)`
        - If `data_format='channels_first'`:
          3D tensor with shape `(batch_size, features, 1)`
    """

    def __init__(self, p: float = 3.0, data_format: Optional[str] = None, keepdims: bool = False, **kwargs) -> None:
        super().__init__(p=p, data_format=data_format, keepdims=keepdims, **kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=4)
        self.gap = keras.layers.GlobalAveragePooling2D(data_format, keepdims)

    def call(self, inputs):
        x = inputs
        if self.data_format == "channels_last":
            mins = tf.math.reduce_min(x, axis=[1, 2])
            x_offset = x - mins[:, tf.newaxis, tf.newaxis, :] + 1
            if self.keepdims:
                mins = mins[:, tf.newaxis, tf.newaxis, :]
        else:
            mins = tf.math.reduce_min(x, axis=[2, 3])
            x_offset = x - mins[:, :, tf.newaxis, tf.newaxis] + 1
            if self.keepdims:
                mins = mins[:, :, tf.newaxis, tf.newaxis]

        #print(x_offset.shape)
        x_offset = self.compute_mean(x_offset)
        x = x_offset + mins - 1
        
        self.add_metric(tf.math.reduce_mean(self.p), "Mean p")
        return x

    def _pos_inf(self, x):
        if self.data_format == "channels_last":
            pool_size = (x.shape[1], x.shape[2])
        else:
            pool_size = (x.shape[2], x.shape[3])
        mpl = keras.layers.MaxPool2D(pool_size=pool_size, data_format=self.data_format)
        x = mpl(x)
        if not self.keepdims:
            if self.data_format == "channels_last":
                x = tf.reshape(x, (x.shape[0], x.shape[3]))
            else:
                x = tf.reshape(x, (x.shape[0], x.shape[1]))
        return x
        