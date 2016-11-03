"""Various improvements to the tensorflow API."""

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages
import functools

def broadcast_as(origin, target, axes=None):
  """Broadcast origin into the shape of target using numpy-style broadcasting.

  If axes is not None, set the shape to be 1 (rather than target.shape[i])
  for each axis i not in axes."""
  in_size = shape_list(origin)
  out_size = shape_list(target)
  result = []
  if axes is None:
    axes = range(len(out_size))
  for d, (i, o) in enumerate(zip(in_size, out_size)):
    if i is None or o is None:
      result.append(1)
    if d in axes:
      assert o % i == 0
      result.append(o//i)
    else:
      result.append(1)
  return tf.tile(origin, result)

def stack(tensor_list, ax):
  """Stack many tensors along a single axis"""
  return tf.concat(ax, [tf.expand_dims(t, ax) for t in tensor_list])

def shape_list(tensor):
  """Return the tensor shape in a form tf.reshape understands."""
  return [x or -1 for x in tensor.get_shape().as_list()]

def safe_squeeze(array, i):
  """Only squeeze a particular axis, and check it was 1"""
  shape = shape_list(array)
  assert shape[i] == 1
  return tf.reshape(array, shape[:i] + (shape[i+1:] if (i+1) else []))

def expand_dims_by_k(array, k):
  """Add k 1s to the end of the tensor's shape"""
  return tf.reshape(array, shape_list(array) + [1]*k)


def fix_batching(f, k, nargs=1):
  """Make a given function f support extra initial dimensions.

  A number of tf.nn operations expect shapes of the form [-1] + lst
  where len(lst) is a fixed constant, and operate independently on the
  -1.  This lets them work on shapes of the form lst2 + lst, where
  lst2 is arbitrary.

  args:
    k: len(lst) that f wants
    nargs: Number of tensors with this property
  """
  @functools.wraps(f)
  def wrapper(*args, **kws):
    arrays = args[:nargs]
    old_shape = shape_list(arrays[0])
    used_shape = old_shape[-k:]
    inputs_reshaped = tuple(tf.reshape(array, [-1]+used_shape)
                       for array in arrays)
    output = f(*(inputs_reshaped + args[nargs:]), **kws)
    new_prefix = old_shape[:-k]
    new_suffix = shape_list(output)[1:]
    output_reshaped = tf.reshape(output, new_prefix + new_suffix)
    return output_reshaped
  return wrapper

softmax = fix_batching(tf.nn.softmax, 1)
conv2d = fix_batching(tf.nn.conv2d, 3)
softmax_cross_entropy_with_logits = fix_batching(tf.nn.softmax_cross_entropy_with_logits, 1, 2)



def masked_moments(x, axes, mask):
    x = x * mask
    num_entries = tf.reduce_sum(tf.ones_like(x) * mask, axes)
    mean = tf.reduce_sum(x, axes) / num_entries
    var = tf.reduce_sum(tf.squared_difference(x, mean)*mask, axes) / num_entries
    return (mean, var)


# From http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
# and https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
def batch_norm(x, phase_train, mask=None, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    x_shape = shape_list(x)
    params_shape = x_shape[-1:]
    BN_DECAY = 0.8
    BN_EPSILON = 1e-3
    with tf.variable_scope(scope) as vs:
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
        moving_mean = tf.get_variable('moving_mean', params_shape,
                                      initializer=tf.zeros_initializer, trainable=False)
        moving_var = tf.get_variable('moving_var', params_shape,
                                     initializer=tf.ones_initializer, trainable=False)
        axes = range(len(x_shape)-1)
        if mask is None:
            batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
        else:
            batch_mean, batch_var = masked_moments(x, axes, mask)

        update_ops = [
            moving_averages.assign_moving_average(moving_mean, batch_mean, BN_DECAY),
            moving_averages.assign_moving_average(moving_var, batch_var, BN_DECAY)]
        def mean_var_with_update():
            with tf.control_dependencies(update_ops):
                return tf.identity(batch_mean), tf.identity(batch_var)

        #mean, var = tf.cond(phase_train,
        #                    mean_var_with_update,
        #                    lambda: (moving_mean, moving_var))
        mean, var = mean_var_with_update()#(batch_mean, batch_var)
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, BN_EPSILON)
    return normed




def print_bn_state(sess, nmaps):
    var_list = 'beta gamma moving_mean moving_var'.split()
    d = {}
    with tf.variable_scope('model/RX1/bn', reuse=True) as vs:
        for v in var_list:
            d[v] = tf.get_variable(v, [nmaps])
    result = sess.run(d, {})
    for v in var_list:
        print(v, result[v])

#numpy.fft.ifft(numpy.conj(numpy.fft.fft(a)) * numpy.fft.fft(b)).round(3)

def softmax_index2d(indices, values, reduce = False):
  indices_shape = shape_list(indices)
  softmax_indices = tf.reshape(
    tf.nn.softmax(
      tf.reshape(indices, [-1, indices_shape[-1]*indices_shape[-2]])),
    indices_shape)
  softmax_indices = tf.complex(softmax_indices, tf.zeros_like(softmax_indices))
  values = tf.complex(values, tf.zeros_like(values))
  fft_of_answer = tf.conj(tf.batch_fft2d(softmax_indices)) * tf.batch_fft2d(values)
  if reduce:
    return tf.reduce_mean(tf.real(tf.batch_ifft(fft_of_answer)), -2)
  else:
    return tf.real(tf.batch_ifft2d(fft_of_answer))

def softmax_index1d(indices, values):
  # indices: bs x height x length 
  # values: stuff x bs x height x length
  indices_shape = shape_list(indices)
  softmax_indices = softmax(indices)
  softmax_indices = tf.complex(softmax_indices, tf.zeros_like(softmax_indices))
  values = tf.complex(values, tf.zeros_like(values))
  fft_of_answer = tf.conj(tf.batch_fft(softmax_indices)) * tf.batch_fft(values)
  return tf.real(tf.batch_ifft(fft_of_answer))
