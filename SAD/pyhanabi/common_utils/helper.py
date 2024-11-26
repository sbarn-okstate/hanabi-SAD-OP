import os
import random
import numpy as np
import tensorflow as tf
from typing import Dict


def get_all_files(root, file_extension):
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(file_extension):
                files.append(os.path.join(folder, f))
    return files


def moving_average(data, period):
    # padding
    left_pad = [data[0] for _ in range(period // 2)]
    right_pad = data[-period // 2 + 1:]
    data = left_pad + data + right_pad
    weights = np.ones(period) / period
    return np.convolve(data, weights, mode="valid")


def mem2str(num_bytes):
    assert num_bytes >= 0
    if num_bytes >= 2 ** 30:  # GB
        val = float(num_bytes) / (2 ** 30)
        result = "%.3f GB" % val
    elif num_bytes >= 2 ** 20:  # MB
        val = float(num_bytes) / (2 ** 20)
        result = "%.3f MB" % val
    elif num_bytes >= 2 ** 10:  # KB
        val = float(num_bytes) / (2 ** 10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result


def sec2str(seconds):
    seconds = int(seconds)
    hour = seconds // 3600
    seconds = seconds % (24 * 3600)
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%dH %02dM %02dS" % (hour, minutes, seconds)


def num2str(n):
    if n < 1e3:
        s = str(n)
        unit = ""
    elif n < 1e6:
        n /= 1e3
        s = "%.3f" % n
        unit = "K"
    else:
        n /= 1e6
        s = "%.3f" % n
        unit = "M"

    s = s.rstrip("0").rstrip(".")
    return s + unit


def get_mem_usage():
    import psutil

    mem = psutil.virtual_memory()
    result = ""
    result += "available: %s, " % (mem2str(mem.available))
    result += "used: %s, " % (mem2str(mem.used))
    result += "free: %s" % (mem2str(mem.free))
    return result


def flatten_first2dim(batch):
    if isinstance(batch, tf.Tensor):
        size = batch.shape[1:]
        batch = tf.reshape(batch, [-1, *size])
        return batch
    elif isinstance(batch, dict):
        return {key: flatten_first2dim(batch[key]) for key in batch}
    else:
        assert False, "unsupported type: %s" % type(batch)


def tensor_slice(t, dim, b, e):
    if isinstance(t, dict):
        return {key: tensor_slice(t[key], dim, b, e) for key in t}
    elif isinstance(t, tf.Tensor):
        return t[b:e]
    else:
        assert False, "Error: unsupported type: %s" % (type(t))


def tensor_index(t, dim, i):
    if isinstance(t, dict):
        return {key: tensor_index(t[key], dim, i) for key in t}
    elif isinstance(t, tf.Tensor):
        return t[i]
    else:
        assert False, "Error: unsupported type: %s" % (type(t))


def one_hot(x, n):
    assert len(x.shape) == 2 and x.shape[1] == 1
    one_hot_x = tf.zeros((x.shape[0], n), dtype=tf.float32)
    one_hot_x = tf.tensor_scatter_nd_update(one_hot_x, tf.reshape(x, (-1, 1)), tf.ones(x.shape[0], dtype=tf.float32))
    return one_hot_x


def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed + 1)
    tf.random.set_seed(rand_seed + 2)


def weights_init(m):
    """custom weights initialization"""
    if isinstance(m, tf.keras.layers.Dense):
        # TensorFlow's equivalent of orthogonal initialization
        initializer = tf.initializers.Orthogonal()
        m.kernel_initializer = initializer
    else:
        print("%s is not custom-initialized." % m.__class__)


def init_net(net, net_file):
    if net_file:
        net.load_weights(net_file)
    else:
        for layer in net.layers:
            weights_init(layer)


def count_output_size(input_shape, model):
    fake_input = tf.random.normal(input_shape)
    output_size = model(fake_input)
    return tf.reduce_prod(output_size.shape)
