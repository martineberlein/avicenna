import logging
from typing import List, Tuple, Optional, Iterable, Set
import itertools

from isla.language import Formula, ConjunctiveFormula
from grammar_graph import gg
from isla import language

from debugging_framework.fuzzingbook.grammar import Grammar
from debugging_framework.input.oracle import OracleResult

from ..data import Input
from .learner import TruthTablePatternCandidateLearner
from .table import Candidate, CandidateSet

logger = logging.getLogger("learner")


class ExhaustivePatternCandidateLearner(
    TruthTablePatternCandidateLearner
):
    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[List[Formula]] = None,
        min_recall: float = 0.9,
        min_specificity: float = 0.6,
    ):
        TruthTablePatternCandidateLearner.__init__(
            self,
            grammar,
            patterns=patterns,
            pattern_file=pattern_file,
            min_recall=min_recall,
            min_precision=min_specificity,
        )
        self.max_conjunction_size = 2
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.exclude_nonterminals: Set[str] = set()
        self.positive_examples_for_learning: List[language.DerivationTree] = []

        # not_patterns = []
        # for pattern in self.patterns:
        #     not_patterns.append(-pattern)

    def learn_candidates(
        self,
        test_inputs: Set[Input],
        exclude_nonterminals: Optional[Iterable[str]] = None,
    ) -> List[Candidate]:
        positive_inputs, negative_inputs = self.categorize_inputs(test_inputs)
        self.update_inputs(positive_inputs, negative_inputs)
        self.exclude_nonterminals = exclude_nonterminals or set()

        candidates = self._learn_invariants(positive_inputs, negative_inputs)
        return candidates

    def _learn_invariants(
        self,
        positive_inputs: Set[Input],
        negative_inputs: Set[Input],
    ) -> List[Candidate]:
        """
        Learn invariants from the positive and negative inputs and return the learned candidates.
        """

        atomic_formulas = self.construct_atomic_candidates(
            self.all_positive_inputs, self.exclude_nonterminals
        )
        new_candidates = {Candidate(formula) for formula in atomic_formulas}

        cans = set(self.candidates.candidates)
        for candidate in new_candidates:
            if candidate not in cans:
                cans.add(candidate)

        for candidate in cans:
            self.evaluate_formula(candidate, positive_inputs, negative_inputs)
        self.get_conjunctions()
        # self.filter_candidates_by_min_requirements()

        return self.sort_candidates()

    def evaluate_formula(
        self,
        candidate: Candidate,
        positive_inputs: Set[Input],
        negative_inputs: Set[Input],
    ):
        """
        Evaluates a formula on a set of inputs.
        """
        if candidate in self.candidates:
            candidate.evaluate(positive_inputs, self.graph)
            if candidate.recall() < self.min_recall:
                # filter out candidates that do not meet the recall threshold
                self.candidates.remove(candidate)
            else:
                # evaluate the candidate on the remaining negative inputs
                candidate.evaluate(negative_inputs, self.graph)
        else:
            candidate.evaluate(self.all_positive_inputs, self.graph)
            candidate.evaluate(self.all_negative_inputs, self.graph)
            if candidate.recall() >= self.min_recall:
                self.candidates.append(candidate)

    def filter_candidates_by_min_requirements(self):
        """
        Filters out candidates that do not meet the minimum requirements.
        """
        candidates_to_remove = [
            candidate
            for candidate in self.candidates
            if candidate.specificity() < self.min_specificity
            or candidate.recall() < self.min_recall
        ]

        for candidate in candidates_to_remove:
            self.candidates.remove(candidate)

    def sort_candidates(self):
        """
        Sorts the candidates based on the sorting strategy.
        """
        sorted_candidates = sorted(
            self.candidates,
            key=lambda candidate: candidate.with_strategy(self.sorting_strategy),
            reverse=True,
        )
        return sorted_candidates

    def get_disjunctions(self):
        """
        Calculate the disjunctions of the formulas.
        """
        pass

    def check_minimum_recall(self, candidates: List[Candidate]) -> bool:
        """
        Check if the recall of the candidates in the combination is greater than the minimum
        """
        return all(candidate.recall() >= self.min_recall for candidate in candidates)

    def is_new_conjunction_valid(
        self, conjunction: Candidate, combination: List[Candidate]
    ) -> bool:
        """
        Check if the new conjunction is valid based on the minimum specificity and the recall of the candidates in
        the combination. The specificity of the new conjunction should be greater than the minimum specificity and
        the specificity of the conjunction should be greater than the specificity of the individual formula.
        """
        new_specificity = conjunction.specificity()
        return new_specificity >= self.min_specificity and all(
            new_specificity > candidate.specificity() for candidate in combination
        )

    def get_conjunctions(
        self,
    ):
        combinations = self.get_possible_conjunctions(self.candidates)

        con_counter = 0
        for combination in combinations:
            # check min recall
            if not self.check_minimum_recall(combination):
                continue
            conjunction: Candidate = combination[0]
            for candidate in combination[1:]:
                conjunction = conjunction & candidate

            conjunction.formula = language.ensure_unique_bound_variables(
                conjunction.formula
            )

            if self.is_new_conjunction_valid(conjunction, combination):
                con_counter += 1
                self.candidates.append(conjunction)

    def get_possible_conjunctions(self, candidate_set: CandidateSet):
        """
        Get all possible conjunctions of the candidate set with a maximum size of max_conjunction_size.
        """
        combinations = []
        candidate_set_without_conjunctions = [
            candidate
            for candidate in candidate_set
            if not isinstance(candidate.formula, ConjunctiveFormula)
        ]
        for level in range(2, self.max_conjunction_size + 1):
            for comb in itertools.combinations(
                candidate_set_without_conjunctions, level
            ):
                combinations.append(comb)
        return combinations

    def reset(self):
        """
        Reset the learner to its initial state.
        """
        self.all_negative_inputs: Set[Input] = set()
        self.all_positive_inputs: Set[Input] = set()
        self.exclude_nonterminals: Set[str] = set()
        self.positive_examples_for_learning: List[language.DerivationTree] = []
        self.candidates = CandidateSet()
        super().reset()
