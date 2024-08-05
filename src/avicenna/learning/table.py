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
from islearn.learner import InvariantLearner

STANDARD_PATTERNS_REPO = "patterns.toml"
logger = logging.getLogger("learner")

from debugging_framework.fuzzingbook.grammar import Grammar
from debugging_framework.input.oracle import OracleResult
from avicenna.input.input import Input


class AvicennaTruthTableRow:
    """
    A truth table row contains a formula, a set of inputs, a list of evaluation results and a combination of inputs and
    evaluation results.
    """

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
    """
    A truth table is a list of truth table rows. It is used to store the results of evaluating formulas on a set of inputs.
    """

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
