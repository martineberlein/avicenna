from typing import Iterable, List, Optional, Set, Tuple
from abc import ABC, abstractmethod

from isla.language import Formula
from debugging_framework.fuzzingbook.grammar import Grammar

from avicenna.input.input import Input
from avicenna.learning.repository import PatternRepository
from avicenna.learning.table import AvicennaTruthTable, AvicennaTruthTableRow


class CandidateLearner(ABC):
    """
    A candidate learner is responsible for learning candidate formulas from a set
    """

    def __init__(self):
        self.candidates: List[Formula] = []

    @abstractmethod
    def learn_candidates(self, test_inputs: Iterable[Input]):
        raise NotImplementedError()

    @abstractmethod
    def get_candidates(self) -> List[Formula]:
        return self.candidates


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
        super().__init__()
        self.grammar = grammar
        if patterns:
            self.patterns: Set[Formula] = set(patterns)
        else:
            self.patterns: Set[Formula] = PatternRepository.from_file(
                pattern_file
            ).get_all()

    @abstractmethod
    def learn_candidates(self, test_inputs: Iterable[Input]) -> List[Tuple[Formula, float, float]]:
        """
        Learn the candidates based on the patterns and the test inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_candidates(self) -> List[Tuple[Formula, float, float]]:
        """
        Get the candidates that have been learned.
        """
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class TruthTablePatternCandidateLearner(PatternCandidateLearner):
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
    ):
        super().__init__(grammar, pattern_file, patterns)
        self.min_precision = min_precision
        self.min_recall = min_recall

        self.precision_truth_table: AvicennaTruthTable = AvicennaTruthTable()
        self.recall_truth_table: AvicennaTruthTable = AvicennaTruthTable()

    def learn_candidates(self, test_inputs: Iterable[Input]) -> List[Tuple[Formula, float, float]]:
        """
        Learn the candidates based on the patterns and the test inputs.
        """
        raise NotImplementedError()

    def meets_criteria(self, precision_value_, recall_value_):
        """
        Checks if the precision and recall values meet the minimum criteria.
        """
        return (
            precision_value_ >= self.min_precision
            and recall_value_ >= self.min_recall
        )

    def get_candidates(self) -> List[Tuple[Formula, float, float]]:
        """
        Returns the all the best formulas (ordered) based on the minimum precision and recall values.
        """
        candidates_with_scores = []

        for idx, precision_row in enumerate(self.precision_truth_table):
            assert isinstance(precision_row, AvicennaTruthTableRow)
            precision_value = 1 - precision_row.eval_result()
            recall_value = self.recall_truth_table[idx].eval_result()

            if self.meets_criteria(precision_value, recall_value):
                candidates_with_scores.append(
                    (precision_row.formula, precision_value, recall_value)
                )

        candidates_with_scores.sort(
            key=lambda x: (x[2], x[1], len(x[0])), reverse=True
        )

        return candidates_with_scores

    def get_best_candidates(
        self,
    ) -> Optional[List[Tuple[Formula, float, float]]]:
        """
        Returns the best formulas (ordered) based on the precision and recall values.
        Thus returns the best of the best formulas.
        """
        candidates_with_scores = self.get_candidates()
        if candidates_with_scores:
            return self._get_best_candidates(candidates_with_scores)

    @staticmethod
    def _get_best_candidates(
        candidates_with_scores: List[Tuple[Formula, float, float]]
    ) -> List[Tuple[Formula, float, float]]:
        """
        Selects the best formulas based on the precision and recall values.
        """
        top_precision, top_recall = (
            candidates_with_scores[0][1],
            candidates_with_scores[0][2],
        )

        return [
            candidate
            for candidate in candidates_with_scores
            if candidate[1] == top_precision and candidate[2] == top_recall
        ]

    def reset(self):
        """
        Resets the precision and recall truth tables. This is useful when the learner is used for multiple runs.
        Minimum precision and recall values are not reset.
        """
        self.precision_truth_table = AvicennaTruthTable()
        self.recall_truth_table = AvicennaTruthTable()
