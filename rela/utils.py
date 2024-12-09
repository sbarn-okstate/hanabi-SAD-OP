"""
Code based on C++ PyTorch code from https://github.com/codeaudit/hanabi_SAD/blob/master/rela/utils.h
"""

import tensorflow as tf
import numpy as np

def get_product(nums):
    prod = 1
    for v in nums:
        prod *= v
    return prod

def push_left(left, vals):
    return [left] + vals

def print_vector(vec):
    print(", ".join(map(str, vec)))

def print_map_key(map_dict):
    print(", ".join(map(str, map_dict.keys())))

def print_map(map_dict):
    for key, value in map_dict.items():
        print(f"{key}: {value}")

def verify_tensors(src, dest):
    if len(src) != len(dest):
        print(f"src.size()[{len(src)}] != dest.size()[{len(dest)}]")
        print("src keys:", list(src.keys()))
        print("dest keys:", list(dest.keys()))
        assert False

    for name, src_tensor in src.items():
        dest_tensor = dest[name]
        if not np.array_equal(src_tensor.shape, dest_tensor.shape):
            print(f"{name}, dstShape: {dest_tensor.shape}, srcShape: {src_tensor.shape}")
            assert False
        if src_tensor.dtype != dest_tensor.dtype:
            print(f"{name}, dstType: {dest_tensor.dtype}, srcType: {src_tensor.dtype}")
            assert False

def copy_tensors(src, dest):
    verify_tensors(src, dest)
    for name, src_tensor in src.items():
        dest[name] = src_tensor.numpy()

def tensor_dict_eq(d0, d1):
    if len(d0) != len(d1):
        return False
    for key, tensor0 in d0.items():
        if not np.array_equal(tensor0, d1[key]):
            return False
    return True

def tensor_dict_index(batch, i):
    return {name: tensor[i] for name, tensor in batch.items()}

def tensor_dict_narrow(batch, dim, i, length, squeeze=False):
    result = {}
    for name, tensor in batch.items():
        narrowed_tensor = tensor[tuple(slice(None) if j != dim else slice(i, i + length) for j in range(tensor.ndim))]
        if squeeze and length == 1:
            narrowed_tensor = np.squeeze(narrowed_tensor, axis=dim)
        result[name] = narrowed_tensor
    return result

def tensor_dict_join(batch, stackdim):
    result = {}
    for name, tensors in batch.items():
        result[name] = np.stack(tensors, axis=stackdim)
    return result

def tensor_dict_clone(input_dict):
    return {name: tensor.copy() for name, tensor in input_dict.items()}

def tensor_dict_zeros_like(input_dict):
    return {name: np.zeros_like(tensor) for name, tensor in input_dict.items()}

def tensor_dict_apply(input_dict, func):
    return {name: func(tensor) for name, tensor in input_dict.items()}

def tensor_vec_dict_append(batch, input_dict):
    for name, tensor in input_dict.items():
        if name not in batch:
            batch[name] = [tensor]
        else:
            batch[name].append(tensor)

def vector_tensor_dict_join(vec, stackdim):
    if not vec:
        return {}
    batch = {name: [] for name in vec[0].keys()}
    for d in vec:
        for name, tensor in d.items():
            batch[name].append(tensor)
    return tensor_dict_join(batch, stackdim)

def i_value_to_tensor_dict(value, device='cpu', detach=False):
    return {key: tensor.numpy() for key, tensor in value.items()}

def tensor_dict_to_tensor_dict(input_dict, device='cpu'):
    return {name: tensor.numpy() for name, tensor in input_dict.items()}
