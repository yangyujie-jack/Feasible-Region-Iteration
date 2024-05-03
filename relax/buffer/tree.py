from typing import Any, Tuple, TypeVar

import numpy as np
import jax, jax.tree_util as tree
from jax import ShapeDtypeStruct

from relax.buffer.base import Buffer
from relax.utils.experience import Experience
from relax.utils.random_utils import seeding

T = TypeVar("T")
ShapeDtypeStructTree = Any  # A tree of same structure as T, but with ShapeDtypeStruct leaves


class TreeBuffer(Buffer[T]):
    def __init__(self, spec: ShapeDtypeStructTree, size: int, seed: int = 0) -> None:
        def create_buffer(sd: ShapeDtypeStruct):
            shape, dtype = (size, *sd.shape), sd.dtype
            return np.empty(shape, dtype=dtype)

        leaves, treedef = tree.tree_flatten(spec)
        self.buffers: Tuple[np.ndarray] = tuple(create_buffer(sd) for sd in leaves)
        self.treedef = treedef

        self.rng, self.seed = seeding(seed)

        self.max_len = size
        self.len = 0
        self.ptr = 0

    def add(self, sample: T, *, from_jax: bool = False) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        leaves = self.treedef.flatten_up_to(sample)
        for leaf, buf in zip(leaves, self.buffers):
            buf[self.ptr] = leaf
        self._advance()

    def add_batch(self, samples: T, *, from_jax: bool = False) -> None:
        if from_jax:
            samples = jax.device_get(samples)
        leaves = self.treedef.flatten_up_to(samples)
        batch_size = leaves[0].shape[0]
        assert batch_size <= self.max_len and all(leaf.shape[0] == batch_size for leaf in leaves)

        start, end = self.ptr, self.ptr + batch_size
        if end > self.max_len:
            # Need to wrap around
            split, remain = self.max_len - start, end - self.max_len
            for leaf, buf in zip(leaves, self.buffers):
                buf[start:] = leaf[:split]
                buf[:remain] = leaf[split:]
        else:
            for leaf, buf in zip(leaves, self.buffers):
                buf[start:end] = leaf

        self._advance(batch_size)

    def sample(self, size: int, *, to_jax: bool = False) -> T:
        indices = self.rng.integers(0, self.len, size=size)
        leaves = tuple(np.take(buf, indices, axis=0) for buf in self.buffers)
        samples = tree.tree_unflatten(self.treedef, leaves)
        if to_jax:
            samples = jax.device_put(samples)
        return samples

    def _advance(self, size: int = 1):
        self.len = min(self.len + size, self.max_len)
        self.ptr = (self.ptr + size) % self.max_len

    def __len__(self):
        return self.len

    def __repr__(self):
        return f"TreeBuffer(size={self.max_len}, len={self.len}, ptr={self.ptr}, treedef={self.treedef})"

    @staticmethod
    def from_example(example: T, size: int, seed: int = 0, remove_batch_dim: bool = True) -> "TreeBuffer[T]":
        def create_shape_dtype_struct(x: np.ndarray):
            if remove_batch_dim:
                assert x.ndim >= 1
                return ShapeDtypeStruct(x.shape[1:], x.dtype)
            else:
                return ShapeDtypeStruct(x.shape, x.dtype)

        spec = tree.tree_map(create_shape_dtype_struct, example)
        return TreeBuffer(spec, size, seed)

    @staticmethod
    def from_experience(obs_dim: int, act_dim: int, size: int, seed: int = 0) -> "TreeBuffer[Experience]":
        example = Experience.create_example(obs_dim, act_dim)
        return TreeBuffer.from_example(example, size, seed, remove_batch_dim=False)
