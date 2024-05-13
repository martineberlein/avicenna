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
from isla.type_defs import Grammar
from islearn.learner import InvariantLearner

STANDARD_PATTERNS_REPO = "patterns.toml"
logger = logging.getLogger("learner")

from debugging_framework.input.oracle import OracleResult
from avicenna.input import Input


class AvicennaTruthTableRow:
    def __init__(
        self,
        formula: language.Formula,
        inputs: Set[Input] = None,
        eval_results: Sequence[bool] = (),
        comb: Dict[Input, bool] = None,
    ):
        self.formula = formula
        self.inputs = inputs or set()
        self.eval_results: List[bool] = list(eval_results)
        self.comb: Dict[Input, bool] = comb or {}

    def __copy__(self):
        return AvicennaTruthTableRow(
            self.formula, self.inputs, self.eval_results, self.comb
        )

    def evaluate(
        self,
        test_inputs: Set[Input],
        graph: gg.GrammarGraph,
    ):
        new_inputs = test_inputs - self.inputs

        for inp in new_inputs:
            eval_result = evaluate(
                self.formula, inp.tree, graph.grammar, graph=graph
            ).is_true()
            self.update_eval_results_and_combination(eval_result, inp)

        self.inputs.update(new_inputs)

    def should_stop_evaluation(
        self, negative_results: int, lazy: bool, result_threshold: float
    ) -> bool:
        return lazy and negative_results > len(self.inputs) * (1 - result_threshold)

    def extend_eval_results_with_false(self):
        self.eval_results += [
            False for _ in range(len(self.inputs) - len(self.eval_results))
        ]

    @staticmethod
    def evaluate_formula_for_input(
        formula: language.Formula, inp: Input, graph: gg.GrammarGraph
    ) -> bool:
        return evaluate(formula, inp.tree, graph.grammar, graph=graph).is_true()

    def update_eval_results_and_combination(self, eval_result: bool, inp: Input):
        self.eval_results.append(eval_result)
        self.comb[inp] = eval_result

    def eval_result(self) -> float:
        assert self.inputs_are_valid()
        return sum(int(entry) for entry in self.eval_results) / len(self.eval_results)

    def inputs_are_valid(self) -> bool:
        return 0 < len(self.inputs) == len(self.eval_results) and all(
            isinstance(entry, bool) for entry in self.eval_results
        )

    def __repr__(self):
        return f"TruthTableRow({str(self.formula)},{repr(self.eval_results)})"

    def __str__(self):
        return f"{self.formula.__str__()}: {', '.join(map(str, self.eval_results))}, {self.comb}"

    def __eq__(self, other):
        return (
            isinstance(other, AvicennaTruthTableRow) and self.formula == other.formula
        )

    def __len__(self):
        return len(self.eval_results)

    def __hash__(self):
        return hash(self.formula)

    def __neg__(self):
        comb = {}
        for inp in self.comb.keys():
            comb[inp] = not self.comb[inp]

        return AvicennaTruthTableRow(
            -self.formula,
            self.inputs,
            [not eval_result for eval_result in self.eval_results],
            comb,
        )

    def __and__(self, other: "AvicennaTruthTableRow") -> "AvicennaTruthTableRow":
        assert len(self.inputs) == len(other.inputs)
        assert len(self.eval_results) == len(other.eval_results)
        assert self.comb.keys() == other.comb.keys()

        eval_result = []
        comb = {}
        for inp in self.comb.keys():
            r = self.comb[inp] and other.comb[inp]
            eval_result.append(r)
            comb[inp] = r

        inputs = copy.copy(self.inputs)

        return AvicennaTruthTableRow(
            self.formula & other.formula, inputs, eval_result, comb
        )

    def __or__(self, other: "AvicennaTruthTableRow") -> "AvicennaTruthTableRow":
        raise NotImplementedError()


class AvicennaTruthTable:
    def __init__(self, rows: Iterable[AvicennaTruthTableRow] = ()):
        self.row_hashes = set()
        self.rows = []
        for row in rows:
            row_hash = hash(row)
            if row_hash not in self.row_hashes:
                self.row_hashes.add(row_hash)
                self.rows.append(row)

    def __deepcopy__(self, memodict=None):
        return AvicennaTruthTable([copy.copy(row) for row in self.rows])

    def __repr__(self):
        return f"TruthTable({repr(self.rows)})"

    def __str__(self):
        return "\n".join(map(str, self.rows))

    def __getitem__(self, item: int | language.Formula) -> AvicennaTruthTableRow:
        if isinstance(item, int):
            return self.rows[item]

        assert isinstance(item, language.Formula)
        try:
            return next(row for row in self.rows if row.formula == item)
        except StopIteration:
            raise KeyError(item)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def append(self, row: AvicennaTruthTableRow):
        row_hash = hash(row)
        if row_hash not in self.row_hashes:
            self.row_hashes.add(row_hash)
            self.rows.append(row)

    def remove(self, row: AvicennaTruthTableRow):
        if hash(row) in self.row_hashes:
            self.rows.remove(row)
            self.row_hashes.remove(hash(row))

    def __add__(self, other: "AvicennaTruthTable") -> "AvicennaTruthTable":
        return AvicennaTruthTable(self.rows + other.rows)

    def __iadd__(self, other: "AvicennaTruthTable") -> "AvicennaTruthTable":
        for row in other.rows:
            self.append(row)

        return self


class PatternLearner:
    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[List[Formula]] = None,
    ):
        pass

    def learn_failure_invariants(
        self,
        test_inputs: Set[Input],
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
        exclude_nonterminals: Optional[Iterable[str]] = None,
    ):
        raise NotImplementedError()


class AviIslearn(InvariantLearner, PatternLearner):
    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[List[Formula]] = None,
        activated_patterns: Optional[Iterable[str]] = None,
        deactivated_patterns: Optional[Iterable[str]] = None,
        min_recall: float = 0.9,
        min_specificity: float = 0.6
    ):
        super().__init__(
            grammar,
            patterns=patterns,
            pattern_file=pattern_file,
            activated_patterns=activated_patterns,
            deactivated_patterns=deactivated_patterns,
            min_recall=min_recall,
            min_specificity=min_specificity
        )
        self.all_negative_inputs: Set[Input] = set()
        self.all_positive_inputs: Set[Input] = set()
        self.initialize_attributes(grammar)

        not_patterns = []
        for pattern in self.patterns:
            not_patterns.append(-pattern)

        # self.patterns.extend(not_patterns)

    def initialize_attributes(self, grammar: Grammar):
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.exclude_nonterminals: Set[str] = set()
        self.positive_examples_for_learning: List[language.DerivationTree] = []

    def learn_failure_invariants(
        self,
        test_inputs: Set[Input],
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
        exclude_nonterminals: Optional[Iterable[str]] = None,
    ):
        positive_inputs, negative_inputs = self.categorize_inputs(test_inputs)
        self.update_inputs(positive_inputs, negative_inputs)
        self.exclude_nonterminals = exclude_nonterminals or set()
        return self._learn_invariants(
            positive_inputs, negative_inputs, precision_truth_table, recall_truth_table
        )

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
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
    ) -> List[Tuple[Formula, float, float]]:
        sorted_positive_inputs = self.sort_and_filter_inputs(self.all_positive_inputs)
        candidates = self.get_candidates(sorted_positive_inputs)

        self.evaluate_recall(candidates, recall_truth_table, positive_inputs)
        self.filter_candidates(precision_truth_table, recall_truth_table)
        self.evaluate_precision(
            precision_truth_table, recall_truth_table, negative_inputs
        )

        self.get_disjunctions()
        self.get_conjunctions(precision_truth_table, recall_truth_table)

        result = self.get_result_list(precision_truth_table, recall_truth_table)
        return result  # , precision_truth_table, recall_truth_table

    @staticmethod
    def clean_up_tables(candidates, precision_truth_table, recall_truth_table):
        rows_to_remove = [
            row for row in recall_truth_table if row.formula not in candidates
        ]
        for row in rows_to_remove:
            recall_truth_table.remove(row)
            precision_truth_table.remove(row)

    def get_candidates(self, sorted_positive_inputs):
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

    def evaluate_recall(
        self, candidates, recall_truth_table: AvicennaTruthTable, positive_inputs
    ):
        logger.info("Evaluating Recall.")
        for candidate in candidates.union(
            set([row.formula for row in recall_truth_table])
        ):
            if (
                len(recall_truth_table) > 0
                and AvicennaTruthTableRow(candidate) in recall_truth_table
            ):
                recall_truth_table[candidate].evaluate(positive_inputs, self.graph)
            else:
                new_row = AvicennaTruthTableRow(candidate)
                new_row.evaluate(self.all_positive_inputs, self.graph)
                recall_truth_table.append(new_row)

    def filter_candidates(
        self,
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
    ):
        # Deleting throws away all calculated evals so far == bad -> maybe only pass TruthTableRows >= self.min_recall?
        rows_to_remove = [
            row
            for row in recall_truth_table
            if row.eval_result() < self.min_recall
            or isinstance(row.formula, ConjunctiveFormula)
        ]
        if self.max_disjunction_size < 2:
            for row in rows_to_remove:
                recall_truth_table.remove(row)
                precision_truth_table.remove(row)

    def evaluate_precision(
        self,
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
        negative_inputs,
    ):
        logger.info("Evaluating Precision.")
        for row in recall_truth_table:
            if len(recall_truth_table) > 0 and row in precision_truth_table:
                precision_truth_table[row.formula].evaluate(negative_inputs, self.graph)
            else:
                # print("Complete Eval Precision")
                new_row = AvicennaTruthTableRow(row.formula)
                new_row.evaluate(self.all_negative_inputs, self.graph)
                precision_truth_table.append(new_row)

        assert len(precision_truth_table) == len(recall_truth_table)

    def get_result_list(
        self, precision_truth_table, recall_truth_table
    ) -> List[Tuple[Formula, float, float]]:
        def meets_criteria(precision_value_, recall_value_):
            return (
                precision_value_ >= self.min_specificity
                and recall_value_ >= self.min_recall
            )

        result = []
        for idx, precision_row in enumerate(precision_truth_table):
            precision_value = 1 - precision_row.eval_result()
            recall_value = recall_truth_table[idx].eval_result()

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
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
    ):
        logger.info("Calculating Boolean Combinations.")
        for level in range(2, self.max_conjunction_size + 1):
            self.process_conjunction_level(
                level, precision_truth_table, recall_truth_table
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


class AvicennaPatternLearner(AviIslearn):
    def _learn_invariants(
        self,
        positive_inputs: Set[Input],
        negative_inputs: Set[Input],
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
    ):
        sorted_positive_inputs = self.sort_and_filter_inputs(self.all_positive_inputs)
        candidates = self.get_candidates(sorted_positive_inputs)
        print(f"Number of candidates: ", len(candidates))

        self.evaluate_recall(candidates, recall_truth_table, positive_inputs)
        self.filter_candidates(precision_truth_table, recall_truth_table)
        self.evaluate_precision(
            precision_truth_table, recall_truth_table, negative_inputs
        )

        dataframe = self.build_dataframe(precision_truth_table, recall_truth_table)
        self.learn_decision_tree(dataframe, precision_truth_table)
        # TODO introduce decTree
        """
         - Dataframe with index columns
         - Pos. Inputs from Recall Table
         - Neg. Inputs from Precision Table
         - Learn decision tree
        """

        result = self.get_result_list(precision_truth_table, recall_truth_table)
        print(len(precision_truth_table), len(recall_truth_table))
        return result  # , precision_truth_table, recall_truth_table

    @staticmethod
    def build_dataframe(
        precision_truth_table: AvicennaTruthTable,
        recall_truth_table: AvicennaTruthTable,
    ):
        dataframe = pandas.DataFrame()

        number_inputs = len(precision_truth_table.rows[0].inputs) + len(
            recall_truth_table.rows[0].inputs
        )
        for idx, row in enumerate(precision_truth_table.rows):
            assert row.formula == recall_truth_table.rows[idx].formula
            dataframe[idx] = pandas.Series([numpy.nan] * number_inputs)
            dataframe[idx] = (
                row.eval_results + recall_truth_table.rows[idx].eval_results
            )

        dataframe["oracle"] = [False] * len(precision_truth_table.rows[0].inputs) + [
            True
        ] * len(recall_truth_table.rows[0].inputs)
        print(dataframe)
        return dataframe

    @staticmethod
    def learn_decision_tree(dataframe: pandas.DataFrame, precision_truth_table):
        from sklearn.tree import DecisionTreeClassifier, export_text
        from isla.language import ISLaUnparser
        from avicenna.treetools import (
            grouped_rules,
            remove_unequal_decisions,
            all_path,
            prediction_for_path,
        )

        sample_bug_count = len(dataframe[(dataframe["oracle"] == True)])
        print("number of bugs found: ", sample_bug_count)
        assert sample_bug_count > 0  # at least one bug triggering sample is required
        sample_count = len(dataframe)
        print(f"Learning with {sample_bug_count} failure inputs of {sample_count}")

        clf = DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=2,  # minimal value
            max_features=5,
            max_depth=2,
            class_weight={
                True: (1.0 / sample_bug_count),
                False: (1.0 / (sample_count - sample_bug_count)),
            },
        )
        clf = clf.fit(dataframe.drop("oracle", axis=1), dataframe["oracle"])

        # clf = remove_unequal_decisions(clf)

        names = [
            ISLaUnparser(precision_truth_table.rows[idx].formula).unparse()
            for idx in dataframe.drop("oracle", axis=1).columns
        ]

        # print(grouped_rules(clf, feature_names=names))
        # print(names)
        tree_text = export_text(clf, feature_names=names)
        # print(tree_text)

        paths = all_path(clf)

        candidates: List[Formula] = []
        for path in paths:
            # print(path, prediction_for_path(clf, path))
            if prediction_for_path(clf, path) == OracleResult.FAILING:
                form: Formula = precision_truth_table.rows[
                    clf.tree_.feature[path[0]]
                ].formula
                assert isinstance(form, Formula)
                for elem in path[1:-1]:
                    new = precision_truth_table.rows[clf.tree_.feature[elem]].formula
                    assert isinstance(new, Formula)
                    form = form.__and__(new)
                    # print("Elem: ", names[clf.tree_.feature[elem]])
                # print(form)
                assert isinstance(form, Formula)
                candidates.append(form)

        for candidate in candidates:
            pass
            print(ISLaUnparser(candidate).unparse())
