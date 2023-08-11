from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')  # Declare a type variable T

class Maybe(Generic[T], ABC):  # Declare that Maybe is generic over T
    @abstractmethod
    def is_just(self) -> bool:
        pass

    @abstractmethod
    def is_nothing(self) -> bool:
        pass

    @abstractmethod
    def value(self) -> T:
        pass

class Just(Maybe[T]):
    def __init__(self, value: T):
        self._value = value

    def is_just(self) -> bool:
        return True

    def is_nothing(self) -> bool:
        return False

    def value(self):
        return self._value

class Nothing(Maybe):
    def is_just(self) -> bool:
        return False

    def is_nothing(self) -> bool:
        return True

    def value(self):
        raise ValueError("No value for Nothing")