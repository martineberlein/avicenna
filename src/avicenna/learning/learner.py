from typing import Iterable, List, Optional, Set, Tuple
from abc import ABC, abstractmethod

from isla.language import Formula
from debugging_framework.fuzzingbook.grammar import Grammar

from avicenna.input.input import Input
from avicenna.learning.repository import PatternRepository
from avicenna.learning.table import AvicennaTruthTable, AvicennaTruthTableRow

from avicenna.learning.candidate import Candidate
from avicenna.learning.metric import FitnessStrategy, RecallPriorityFitness


class CandidateLearner(ABC):
    """
    A candidate learner is responsible for learning candidate formulas from a set
    """

    def __init__(self):
        self.candidates: List[Candidate] = []

    @abstractmethod
    def learn_candidates(
        self, test_inputs: Iterable[Input]
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
        sorting_strategy: FitnessStrategy = RecallPriorityFitness(),
    ):
        super().__init__(grammar, pattern_file, patterns)
        self.min_precision = min_precision
        self.min_recall = min_recall

        self.precision_truth_table: AvicennaTruthTable = AvicennaTruthTable()
        self.recall_truth_table: AvicennaTruthTable = AvicennaTruthTable()
        self.sorting_strategy = sorting_strategy

    def meets_minimum_criteria(self, precision_value_, recall_value_):
        """
        Checks if the precision and recall values meet the minimum criteria.
        :param precision_value_: The precision value.
        :param recall_value_: The recall value.
        """
        return (
            precision_value_ >= self.min_precision and recall_value_ >= self.min_recall
        )

    def get_candidates(self) -> Optional[List[Candidate]]:
        """
        Returns the all the best formulas (ordered) based on the minimum precision and recall values.
        :return Optional[List[Candidate]]: The learned candidates.
        """
        candidates: List[Candidate] = []

        for idx, precision_row in enumerate(self.precision_truth_table):
            assert isinstance(precision_row, AvicennaTruthTableRow)
            precision_value = 1 - precision_row.eval_result()
            recall_value = self.recall_truth_table[idx].eval_result()

            if self.meets_minimum_criteria(precision_value, recall_value):
                candidates.append(
                    Candidate(precision_row.formula, precision_value, recall_value)
                )

        return sorted(
            candidates,
            key=lambda c: c.with_strategy(self.sorting_strategy),
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

    @staticmethod
    def _get_best_candidates(candidates: List[Candidate]) -> List[Candidate]:
        """
        Selects the best formulas based on the precision and recall values.
        :param candidates: The candidates to select the best from.
        :return List[Candidate]: The best learned candidates.
        """
        return [candidate for candidate in candidates if candidate == candidates[0]]

    def reset(self):
        """
        Resets the precision and recall truth tables. This is useful when the learner is used for multiple runs.
        Minimum precision and recall values are not reset.
        """
        self.precision_truth_table = AvicennaTruthTable()
        self.recall_truth_table = AvicennaTruthTable()
