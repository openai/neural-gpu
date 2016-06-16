"""Various improvements to the tensorflow API."""
import tensorflow as tf
import functools

def shape_list(tensor):
  """Return the tensor shape in a form tf.reshape understands."""
  return [x or -1 for x in tensor.get_shape().as_list()]

def safe_squeeze(array, i):
  shape = shape_list(array)
  assert shape[i] == 1
  return tf.reshape(array, shape[:i] + (shape[i+1:] if (i+1) else []))

def expand_dims_by_k(array, k):
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

