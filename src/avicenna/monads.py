from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Any, Type
from dataclasses import dataclass


T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E", bound=Exception)
R = TypeVar("R")
S = TypeVar("S")


@dataclass
class Monad(ABC, Generic[T]):
    a: T

    @abstractmethod
    def bind(self, f: Callable[[T], "Monad[S]"]) -> "Monad[S]":
        raise NotImplementedError()


class Maybe(Generic[T], ABC):
    @abstractmethod
    def is_just(self) -> bool:
        pass

    @abstractmethod
    def is_nothing(self) -> bool:
        pass

    def map(self, func) -> T:
        raise NotImplementedError()

    def bind(self, func):
        raise NotImplementedError()

    @abstractmethod
    def value(self) -> T:
        pass

    @classmethod
    def just(cls, value):
        return Just(value)

    @classmethod
    def nothing(cls):
        return Nothing()


class Just(Maybe[T]):
    def __init__(self, value: T):
        self._value = value

    def map(self, func: Callable[[T], U]) -> "Maybe[U]":
        return Just(func(self._value))

    def bind(self, func: Callable[[T], "Maybe[U]"]) -> "Maybe[U]":
        return func(self._value)

    def is_just(self) -> bool:
        return True

    def is_nothing(self) -> bool:
        return False

    def value(self):
        return self._value


class Nothing(Maybe):
    def map(self, func: Callable[[T], U]) -> "Maybe[U]":
        return self

    def bind(self, func: Callable[[T], "Maybe[U]"]) -> "Maybe[U]":
        return self

    def is_just(self) -> bool:
        return False

    def is_nothing(self) -> bool:
        return True

    def value(self):
        raise ValueError("No value for Nothing")


@dataclass
class Exceptional(Generic[E, T]):
    @staticmethod
    def of(f: Callable[[], T]) -> "Exceptional[E, T]":
        try:
            return Success(f())
        except Exception as exc:
            return Failure(exc)

    @abstractmethod
    def get(self) -> T:
        pass

    @abstractmethod
    def map(self, f: Callable[[T], S]) -> "Exceptional[S]":
        pass

    @abstractmethod
    def bind(self, f: Callable[[T], "Exceptional[E, T]"]) -> "Exceptional[E, T]":
        raise NotImplementedError()

    @abstractmethod
    def recover(self, f: Callable[[E], T], *exc_types: Type[E]) -> "Exceptional[E, T]":
        pass

    @abstractmethod
    def reraise(self) -> "Exceptional[T]":
        pass


@dataclass
class Success(Generic[T], Exceptional[Exception, T]):
    a: T

    def get(self) -> T:
        return self.a

    def bind(self, f: Callable[[T], "Exceptional[S]"]) -> "Exceptional[S]":
        return f(self.a)

    def map(self, f: Callable[[T], S]) -> "Exceptional[S]":
        return Exceptional.of(lambda: f(self.a))

    def recover(self, _, *__) -> "Success[T]":
        return self

    def reraise(self) -> "Success[T]":
        return self


@dataclass
class Failure(Generic[E], Exceptional[E, Any]):
    a: E

    def get(self) -> E:
        raise AttributeError(f"{type(self).__name__} does not support get()")

    def bind(self, _) -> "Exceptional[T]":
        return self

    def map(self, _) -> "Exceptional[S]":
        return self

    def recover(self, f: Callable[[E], T], *exc_types: Type[E]) -> "Exceptional[E, T]":
        if not exc_types or any(isinstance(self.a, exc) for exc in exc_types):
            return Exceptional.of(lambda: f(self.a))
        else:
            return self

    def reraise(self) -> Exceptional[E, Any]:
        raise self.a


def check_empty(x: T) -> Exceptional[Exception, T]:
    if x is None:
        return Failure(AssertionError())
    elif isinstance(x, (str, list, dict, set, tuple)) and not x:
        return Failure(AssertionError())
    return Success(x)
