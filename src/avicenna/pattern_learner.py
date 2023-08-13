import copy
import logging
import functools
from typing import List, Tuple, Dict, Optional, Iterable, Sequence, Set
import itertools

from isla.evaluator import evaluate
from isla.language import Formula, ConjunctiveFormula
from islearn.learner import weighted_geometric_mean
from grammar_graph import gg
from isla import language
from isla.type_defs import Grammar
from islearn.learner import InvariantLearner

STANDARD_PATTERNS_REPO = "patterns.toml"
logger = logging.getLogger("learner")

from avicenna.input import Input
from avicenna.oracle import OracleResult


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


class AviIslearn(InvariantLearner):
    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        activated_patterns: Optional[Iterable[str]] = None,
        deactivated_patterns: Optional[Iterable[str]] = None,
    ):
        super().__init__(
            grammar,
            pattern_file=pattern_file,
            activated_patterns=activated_patterns,
            deactivated_patterns=deactivated_patterns,
        )
        self.all_negative_inputs: Set[Input] = set()
        self.all_positive_inputs: Set[Input] = set()
        self.initialize_attributes(grammar)
        print(self.patterns)

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
        positive_inputs = {inp for inp in test_inputs if inp.oracle == OracleResult.BUG}
        negative_inputs = {
            inp for inp in test_inputs if inp.oracle == OracleResult.NO_BUG
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
    ):
        sorted_positive_inputs = self.sort_and_filter_inputs(self.all_positive_inputs)
        candidates = self.get_candidates(sorted_positive_inputs)

        self.evaluate_recall(candidates, recall_truth_table, positive_inputs)
        self.filter_candidates(precision_truth_table, recall_truth_table)
        self.evaluate_precision(
            precision_truth_table, recall_truth_table, negative_inputs
        )
        self.get_disjunctions()
        self.get_conjunctions(precision_truth_table, recall_truth_table)

        result = self.get_result_dict(precision_truth_table, recall_truth_table)
        return result #, precision_truth_table, recall_truth_table

    @staticmethod
    def clean_up_tables(candidates, precision_truth_table, recall_truth_table):
        rows_to_remove = [
            row
            for row in recall_truth_table
            if row.formula not in candidates
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

    def get_result_dict(
        self, precision_truth_table, recall_truth_table
    ) -> Dict[Formula, Tuple[float, float]]:
        def meets_criteria(precision_value_, recall_value_):
            return (
                precision_value_ >= self.min_specificity
                and recall_value_ >= self.min_recall
            )

        result = {}
        for idx, precision_row in enumerate(precision_truth_table):
            precision_value = 1 - precision_row.eval_result()
            recall_value = recall_truth_table[idx].eval_result()

            if meets_criteria(precision_value, recall_value):
                result[precision_row.formula] = (precision_value, recall_value)

        sorted_result = dict(
            sorted(result.items(), key=lambda p: (p[1], -len(p[0])), reverse=True)
        )
        logger.info(
            "Found %d invariants with precision >= %d%% and recall >= %d%%.",
            len(
                [p for p in result.values()]
            ),  # if p[0] >= self.min_specificity and p[1] >= self.min_recall]),
            int(self.min_specificity * 100),
            int(self.min_recall * 100),
        )

        return sorted_result

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
        return not any(
            recall_truth_table[idx].eval_result() < self.min_recall
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
