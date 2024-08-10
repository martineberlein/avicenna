from abc import ABC, abstractmethod
from typing import Dict, Set, Union, List
from collections import defaultdict

from ..data import Input


class TResultMonad:
    def __init__(self, value):
        self._value = (value, None) if not isinstance(value, tuple) else value

    def map(self, func):
        return TResultMonad(func(*self._value))

    def value(self):
        return self._value


class Failure:
    def __init__(self, exception: Exception):
        self.exception = exception
        self.message = str(exception)

    def __hash__(self):
        return hash(type(self.exception)) + hash(self.message)

    def __eq__(self, other):
        if not isinstance(other, Failure):
            return False
        return (
            isinstance(other.exception, type(self.exception))
            and other.message == self.message
        )

    def __repr__(self):
        if self.message:
            return f"{type(self.exception).__name__}: {self.message}"
        else:
            return f"{type(self.exception).__name__}"

    def __str__(self):
        return self.__repr__()


class Report(ABC):
    def __init__(self, name: str = "EvoGFuzz"):
        self.failures: Dict[Failure, Set[Input]] = defaultdict(set)
        self.name = name

    def __repr__(self):
        report = f"Report for {self.name}\n"
        report += (
            f"Found {len(self.get_all_failing_inputs())} failure-inducing"
            f" inputs ({len(self.failures.keys())} Exceptions):\n"
        )
        report += "\n".join(
            f"{failure}: {len(self.failures[failure])}" for failure in self.failures
        )
        return report

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def add_failure(
        self, test_input: Input, failure: Union[Exception, Failure], **kwargs
    ):
        raise NotImplementedError

    def get_failures(self) -> Dict[Failure, Set[Input]]:
        return self.failures

    def get_all_failing_inputs(self) -> List[Input]:
        flat = []
        for v in self.failures.values():
            flat.extend(list(v))
        return flat


class SingleFailureReport(Report):
    def add_failure(self, test_input: Input, failure=None, **kwargs):
        self.failures[Failure(Exception())].add(test_input)


class MultipleFailureReport(Report):
    def add_failure(
        self, test_input: Input, failure: Union[Exception, Failure], **kwargs
    ):
        if isinstance(failure, Exception):
            failure = Failure(failure)

        self.failures[failure].add(test_input)
