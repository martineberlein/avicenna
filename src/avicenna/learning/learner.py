from typing import Iterable, List, Optional, Set
from abc import ABC, abstractmethod

from isla.language import Formula
from debugging_framework.fuzzingbook.grammar import Grammar

from avicenna.input.input import Input
from avicenna.learning.repository import PatternRepository


class CandidateLearner(ABC):
    """
    A candidate learner is responsible for learning candidate formulas from a set
    """

    def __init__(self):
        self.candidates: List[Formula] = []

    @abstractmethod
    def learn_candidates(self, test_inputs: Iterable[Input]):
        raise NotImplementedError()

    def get_candidates(self) -> List[Formula]:
        return self.candidates


class PatternCandidateLearner(CandidateLearner):
    """
    A candidate learner that learns formulas based on patterns from a pattern repository
    """

    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[Iterable[Formula]] = None,
    ):
        super().__init__()
        self.grammar = grammar
        if patterns:
            self.patterns: Set[Formula] = set(patterns)
        else:
            self.patterns: Set[Formula] = PatternRepository.from_file(
                pattern_file
            ).get_all()

    def learn_candidates(self, test_inputs: Iterable[Input]):
        raise NotImplementedError()
