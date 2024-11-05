from typing import Iterable, List, Optional, Set
from abc import ABC, abstractmethod

from isla.language import Formula
from debugging_framework.fuzzingbook.grammar import Grammar

from ..data import Input
from .repository import PatternRepository
from .table import Candidate, CandidateSet
from .metric import FitnessStrategy, RecallPriorityLengthFitness
from .constructor import AtomicFormulaInstantiation


class CandidateLearner(ABC):
    """
    A candidate learner is responsible for learning candidate formulas from a set
    """

    def __init__(self):
        self.candidates: List[Candidate] = []

    @abstractmethod
    def learn_candidates(
        self, test_inputs: Iterable[Input], **kwargs
    ) -> Optional[List[Candidate]]:
        """
        Learn the candidates based on the test inputs.
        :param test_inputs: The test inputs to learn the candidates from.
        :return Optional[List[Candidate]]: The learned candidates.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_candidates(self) -> Optional[List[Candidate]]:
        """
        Get the candidates Formula that have been learned.
        :return Optional[List[Candidate]]: The learned candidates.
        """
        return self.candidates

    @abstractmethod
    def get_best_candidates(self) -> Optional[List[Candidate]]:
        """
        Get the best candidates that have been learned.
        :return Optional[List[Candidate]]: The best learned candidates.
        """
        raise NotImplementedError()


class PatternCandidateLearner(CandidateLearner, ABC):
    """
    A candidate learner that learns formulas based on patterns from a pattern repository
    """

    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[Iterable[Formula]] = None,
        use_fast_evaluation: bool = False,
    ):
        """
        Initialize the pattern candidate learner with a grammar and a pattern file or patterns.
        :param grammar: The grammar to use for the formulas.
        :param pattern_file: The file containing the patterns.
        :param patterns: The patterns to use.
        """
        super().__init__()
        self.grammar = grammar
        if patterns:
            self.patterns: Set[Formula] = set(patterns)
        else:
            self.patterns: Set[Formula] = PatternRepository.from_file(
                pattern_file
            ).get_all()

        self.atomic_candidate_constructor = AtomicFormulaInstantiation(
            grammar, list(self.patterns)
        )
        self.use_fast_evaluation = use_fast_evaluation

    def construct_atomic_candidates(self, positive_inputs: Set[Input], exclude_non_terminals: Set[str] = None) -> Set[Formula]:
        """
        Construct the atomic candidates based on the patterns.
        :param positive_inputs: The positive inputs to construct the candidates from.
        :param exclude_non_terminals: The non-terminals to exclude from the candidates.
        :return Set[Formula]: The atomic formula candidates.
        """
        candidates = self.atomic_candidate_constructor.construct_candidates(
            positive_inputs, exclude_non_terminals
        )
        return candidates


class TruthTablePatternCandidateLearner(PatternCandidateLearner, ABC):
    """
    A candidate learner that learns formulas based on patterns from a pattern repository
    """

    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[Iterable[Formula]] = None,
        min_precision: float = 0.6,
        min_recall: float = 0.9,
        sorting_strategy: FitnessStrategy = RecallPriorityLengthFitness(),
        use_fast_evaluation: bool = False,
    ):
        super().__init__(grammar, pattern_file, patterns, use_fast_evaluation)
        self.min_specificity = min_precision
        self.min_recall = min_recall

        self.candidates: CandidateSet = CandidateSet()
        self.sorting_strategy = sorting_strategy

    def meets_minimum_criteria(self, precision_value_, recall_value_):
        """
        Checks if the precision and recall values meet the minimum criteria.
        :param precision_value_: The precision value.
        :param recall_value_: The recall value.
        """
        return (
            precision_value_ >= self.min_specificity
            and recall_value_ >= self.min_recall
        )

    def get_candidates(self) -> Optional[List[Candidate]]:
        """
        Returns the all the best formulas (ordered) based on the minimum precision and recall values.
        :return Optional[List[Candidate]]: The learned candidates.
        """

        return sorted(
            self.candidates,
            key=lambda c: self.sorting_strategy.evaluate(c),
            reverse=True,
        )

    def get_best_candidates(
        self,
    ) -> Optional[List[Candidate]]:
        """
        Returns the best formulas (ordered) based on the precision and recall values.
        Thus returns the best of the best formulas.
        :return Optional[List[Candidate]]: The best learned candidates.
        """
        candidates = self.get_candidates()
        if candidates:
            return self._get_best_candidates(candidates)

    def _get_best_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Selects the best formulas based on the precision and recall values.
        :param candidates: The candidates to select the best from.
        :return List[Candidate]: The best learned candidates.
        """
        return [
            candidate
            for candidate in candidates
            if self.sorting_strategy.is_equal(candidate, candidates[0])
        ]

    def reset(self):
        """
        Resets the precision and recall truth tables. This is useful when the learner is used for multiple runs.
        Minimum precision and recall values are not reset.
        """
        self.candidates = CandidateSet()
