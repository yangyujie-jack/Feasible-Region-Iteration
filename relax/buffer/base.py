from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Buffer(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def add(self, sample: T, *, from_jax: bool = False) -> None:
        ...

    @abstractmethod
    def add_batch(self, samples: T, *, from_jax: bool = False) -> None:
        ...

    @abstractmethod
    def sample(self, size: int, *, to_jax: bool = False) -> T:
        ...
