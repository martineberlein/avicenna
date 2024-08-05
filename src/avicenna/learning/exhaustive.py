import copy
import logging
import functools
from typing import List, Tuple, Dict, Optional, Iterable, Sequence, Set
import itertools

import pandas
import numpy
from isla.evaluator import evaluate
from isla.language import Formula, ConjunctiveFormula
from islearn.learner import weighted_geometric_mean
from grammar_graph import gg
from isla import language
from debugging_framework.fuzzingbook.grammar import Grammar
from islearn.learner import InvariantLearner

from debugging_framework.input.oracle import OracleResult
from avicenna.input.input import Input

from avicenna.learning.learner import PatternCandidateLearner
from avicenna.learning.table import AvicennaTruthTable, AvicennaTruthTableRow

logger = logging.getLogger("learner")


class ExhaustivePatternCandidateLearner(PatternCandidateLearner, InvariantLearner):
    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[List[Formula]] = None,
        min_recall: float = 0.9,
        min_specificity: float = 0.6,
    ):
        PatternCandidateLearner.__init__(
            self, grammar, patterns=patterns, pattern_file=pattern_file
        )
        InvariantLearner.__init__(self, grammar, patterns=list(self.patterns))
        self.min_recall = min_recall
        self.min_specificity = min_specificity
        self.all_negative_inputs: Set[Input] = set()
        self.all_positive_inputs: Set[Input] = set()
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.exclude_nonterminals: Set[str] = set()
        self.positive_examples_for_learning: List[language.DerivationTree] = []

        self.precision_truth_table: AvicennaTruthTable = AvicennaTruthTable()
        self.recall_truth_table: AvicennaTruthTable = AvicennaTruthTable()

        not_patterns = []
        for pattern in self.patterns:
            not_patterns.append(-pattern)

    def learn_candidates(
        self,
        test_inputs: Set[Input],
        exclude_nonterminals: Optional[Iterable[str]] = None,
    ) -> List[Tuple[Formula, float, float]]:
        positive_inputs, negative_inputs = self.categorize_inputs(test_inputs)
        self.update_inputs(positive_inputs, negative_inputs)
        self.exclude_nonterminals = exclude_nonterminals or set()

        candidates = self._learn_invariants(positive_inputs, negative_inputs)
        return candidates

    @staticmethod
    def categorize_inputs(test_inputs: Set[Input]) -> Tuple[Set[Input], Set[Input]]:
        positive_inputs = {
            inp for inp in test_inputs if inp.oracle == OracleResult.FAILING
        }
        negative_inputs = {
            inp for inp in test_inputs if inp.oracle == OracleResult.PASSING
        }
        return positive_inputs, negative_inputs

    def update_inputs(self, positive_inputs: Set[Input], negative_inputs: Set[Input]):
        self.all_positive_inputs.update(positive_inputs)
        self.all_negative_inputs.update(negative_inputs)

    def _learn_invariants(
        self,
        positive_inputs: Set[Input],
        negative_inputs: Set[Input],
    ) -> List[Tuple[Formula, float, float]]:
        sorted_positive_inputs = self.sort_and_filter_inputs(self.all_positive_inputs)
        candidates = self.get_candidates(sorted_positive_inputs)

        self.evaluate_recall(candidates, positive_inputs)
        self.filter_candidates()
        self.evaluate_precision(negative_inputs)

        self.get_disjunctions()
        self.get_conjunctions()

        result = self.get_result_list()
        return result  # , precision_truth_table, recall_truth_table

    @staticmethod
    def clean_up_tables(candidates, precision_truth_table, recall_truth_table):
        rows_to_remove = [
            row for row in recall_truth_table if row.formula not in candidates
        ]
        for row in rows_to_remove:
            recall_truth_table.remove(row)
            precision_truth_table.remove(row)

    def get_candidates(self, sorted_positive_inputs) -> Set[Formula]:
        candidates = self.generate_candidates(
            self.patterns, [inp.tree for inp in sorted_positive_inputs]
        )
        logger.info("Found %d invariant candidates.", len(candidates))
        return candidates

    def sort_and_filter_inputs(
        self,
        positive_inputs: Set[Input],
        max_number_positive_inputs_for_learning: int = 10,
    ):
        p_dummy = copy.deepcopy(positive_inputs)
        sorted_positive_inputs = self._sort_inputs(
            p_dummy,
            self.filter_inputs_for_learning_by_kpaths,
            more_paths_weight=1.7,
            smaller_inputs_weight=1.0,
        )
        logger.info(
            "Keeping %d positive examples for candidate generation.",
            len(sorted_positive_inputs[:max_number_positive_inputs_for_learning]),
        )

        return sorted_positive_inputs[:max_number_positive_inputs_for_learning]

    def evaluate_recall(self, candidates, positive_inputs):
        logger.info("Evaluating Recall.")
        for candidate in candidates.union(
            set([row.formula for row in self.recall_truth_table])
        ):
            if (
                len(self.recall_truth_table) > 0
                and AvicennaTruthTableRow(candidate) in self.recall_truth_table
            ):
                self.recall_truth_table[candidate].evaluate(positive_inputs, self.graph)
            else:
                new_row = AvicennaTruthTableRow(candidate)
                new_row.evaluate(self.all_positive_inputs, self.graph)
                self.recall_truth_table.append(new_row)

    def filter_candidates(
        self,
    ):
        # Deleting throws away all calculated evals so far == bad -> maybe only pass TruthTableRows >= self.min_recall?
        rows_to_remove = [
            row
            for row in self.recall_truth_table
            if row.eval_result() < self.min_recall
            or isinstance(row.formula, ConjunctiveFormula)
        ]
        if self.max_disjunction_size < 2:
            for row in rows_to_remove:
                self.recall_truth_table.remove(row)
                self.precision_truth_table.remove(row)

    def evaluate_precision(
        self,
        negative_inputs,
    ):
        logger.info("Evaluating Precision.")
        for row in self.recall_truth_table:
            if len(self.recall_truth_table) > 0 and row in self.precision_truth_table:
                self.precision_truth_table[row.formula].evaluate(
                    negative_inputs, self.graph
                )
            else:
                # print("Complete Eval Precision")
                new_row = AvicennaTruthTableRow(row.formula)
                new_row.evaluate(self.all_negative_inputs, self.graph)
                self.precision_truth_table.append(new_row)

        assert len(self.precision_truth_table) == len(self.recall_truth_table)

    def get_result_list(
        self,
    ) -> List[Tuple[Formula, float, float]]:
        def meets_criteria(precision_value_, recall_value_):
            return (
                precision_value_ >= self.min_specificity
                and recall_value_ >= self.min_recall
            )

        result = []
        for idx, precision_row in enumerate(self.precision_truth_table):
            precision_value = 1 - precision_row.eval_result()
            recall_value = self.recall_truth_table[idx].eval_result()

            if meets_criteria(precision_value, recall_value):
                result.append((precision_row.formula, precision_value, recall_value))

        result.sort(key=lambda x: (x[1], x[2], -len(x[0])), reverse=True)

        logger.info(
            "Found %d invariants with precision >= %d%% and recall >= %d%%.",
            len(
                result
            ),  # if p[0] >= self.min_specificity and p[1] >= self.min_recall]),
            int(self.min_specificity * 100),
            int(self.min_recall * 100),
        )
        return result

    def get_disjunctions(self):
        pass

    def get_conjunctions(
        self,
    ):
        logger.info("Calculating Boolean Combinations.")
        for level in range(2, self.max_conjunction_size + 1):
            self.process_conjunction_level(
                level, self.precision_truth_table, self.recall_truth_table
            )

    def process_conjunction_level(
        self,
        level: int,
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
    ):
        combinations = self.get_combinations_of_truth_table_rows(
            level, precision_truth_table
        )

        for rows_with_indices in combinations:
            self.process_combination(
                rows_with_indices, precision_truth_table, recall_truth_table
            )

    @staticmethod
    def get_combinations_of_truth_table_rows(
        level: int, truth_table: AvicennaTruthTable
    ):
        return itertools.combinations(enumerate(truth_table), level)

    def process_combination(
        self,
        rows_with_indices,
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
    ):
        precision_table_rows = [row for (_, row) in rows_with_indices]

        if not self.rows_meet_minimum_recall(rows_with_indices, recall_truth_table):
            return

        self.add_conjunction_to_truth_table(
            precision_table_rows,
            precision_truth_table,
            recall_truth_table,
            rows_with_indices,
        )

    def rows_meet_minimum_recall(
        self, rows_with_indices, recall_truth_table: AvicennaTruthTable
    ) -> bool:
        return any(
            recall_truth_table[idx].eval_result() >= self.min_recall
            for idx, _ in rows_with_indices
        )

    def rows_meet_minimum_precision(
        self, rows_with_indices, precision_truth_table: AvicennaTruthTable
    ) -> bool:
        return not any(
            precision_truth_table[idx].eval_result() < self.min_specificity
            for idx, _ in rows_with_indices
        )

    def add_conjunction_to_truth_table(
        self,
        precision_table_rows,
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
        rows_with_indices,
    ):
        precision_conjunction = self.get_conjunction(precision_table_rows)
        recall_conjunction = self.get_conjunction(
            [recall_truth_table[idx] for idx, _ in rows_with_indices]
        )

        if self.is_new_conjunction_valid(precision_conjunction, precision_table_rows):
            precision_truth_table.append(precision_conjunction)
            recall_truth_table.append(recall_conjunction)

    @staticmethod
    def get_conjunction(table_rows) -> AvicennaTruthTableRow:
        conjunction = functools.reduce(AvicennaTruthTableRow.__and__, table_rows)
        conjunction.formula = language.ensure_unique_bound_variables(
            conjunction.formula
        )
        return conjunction

    def is_new_conjunction_valid(
        self, conjunction: AvicennaTruthTableRow, precision_table_rows
    ) -> bool:
        new_precision = 1 - conjunction.eval_result()
        return new_precision >= self.min_specificity and all(
            new_precision > 1 - row.eval_result() for row in precision_table_rows
        )

    def _sort_inputs(
        self,
        inputs: Set[Input],
        filter_inputs_for_learning_by_kpaths: bool,
        more_paths_weight: float = 1.0,
        smaller_inputs_weight: float = 0.0,
    ) -> List[Input]:
        assert more_paths_weight or smaller_inputs_weight
        result: List[Input] = []

        tree_paths = {
            inp: {
                path
                for path in self.graph.k_paths_in_tree(inp.tree.to_parse_tree(), self.k)
                if (
                    not isinstance(path[-1], gg.TerminalNode)
                    or (
                        not isinstance(path[-1], gg.TerminalNode)
                        and len(path[-1].symbol) > 1
                    )
                )
            }
            for inp in inputs
        }

        covered_paths: Set[Tuple[gg.Node, ...]] = set([])
        max_len_input = max(len(inp.tree) for inp in inputs)

        def uncovered_paths(inp: Input) -> Set[Tuple[gg.Node, ...]]:
            return {path for path in tree_paths[inp] if path not in covered_paths}

        def sort_by_paths_key(inp: Input) -> float:
            return len(uncovered_paths(inp))

        def sort_by_length_key(inp: Input) -> float:
            return len(inp.tree)

        def sort_by_paths_and_length_key(inp: Input) -> float:
            return weighted_geometric_mean(
                [len(uncovered_paths(inp)), max_len_input - len(inp.tree)],
                [more_paths_weight, smaller_inputs_weight],
            )

        if not more_paths_weight:
            key = sort_by_length_key
        elif not smaller_inputs_weight:
            key = sort_by_paths_key
        else:
            key = sort_by_paths_and_length_key

        while inputs:
            inp = sorted(inputs, key=key, reverse=True)[0]
            inputs.remove(inp)
            uncovered = uncovered_paths(inp)

            if filter_inputs_for_learning_by_kpaths and not uncovered:
                continue

            covered_paths.update(uncovered)
            result.append(inp)

        return result

    def reduce_inputs(
        self, test_inputs: Set[Input], negative_inputs: Set[Input]
    ) -> Tuple[Set[Input], Set[Input]]:
        raise NotImplementedError()
