import copy
from typing import Iterable, Set

from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from pathos import multiprocessing as pmp

from avicenna.input import Input
from avicenna.oracle import OracleResult

"""
    Idea:
    1. Create Table
        2. Of Rows with formulas (unique)
        3. Evaluate each row (formula)
            4. however only with new inputs
            5. safe tp, fp, fn, tn
        Since all inputs are unique and no inputs will be evaluated twice this should work 
"""


class TruthTableRow:
    def __init__(
        self,
        formula: language.Formula,
        n: int = 0,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
        tn: int = 0,
    ):
        self.formula = formula
        self.n = n
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def __copy__(self):
        return TruthTableRow(self.formula, self.n, self.tp, self.fp, self.fn, self.tn)

    def evaluate(self, inputs: Set[Input], graph: gg.GrammarGraph) -> "TruthTableRow":
        """If lazy is True, then the evaluation stops as soon as result_threshold can no longer be
        reached. E.g., if result_threshold is .9 and there are 100 inputs, then after more than
        10 negative results, 90% positive results is no longer possible."""

        for inp in inputs:
            eval_result = evaluate(
                self.formula, inp.tree, graph.grammar, graph=graph
            ).is_true()

            bool_oracle = (True if inp.oracle == OracleResult.BUG else False)

            if eval_result is True and bool_oracle is True:
                    self.tp += 1
            elif eval_result is True and bool_oracle is False:
                self.fp += 1
            elif eval_result is False and bool_oracle is True:
                self.fn += 1
            elif eval_result is False and bool_oracle is False:
                self.tn += 1
            else:
                raise Exception("This should not be possible")

            self.n += 1

        return self

    def eval_result(self):
        assert self.n == (self.tp + self.tn + self.fn + self.fp)
        return self.n, self.tp, self.fp, self.fn, self.tn

    def __repr__(self):
        return f"TruthTableRow({repr(self.formula)})"

    def __str__(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1 = (2 * precision * recall) / (precision + recall)
        return f"{self.formula}: #inputs: {self.n};" \
               f" precision {round(precision * 100, 3)}%;" \
               f" recall {round(recall * 100, 3)}%;" \
               f" f1 {round(f1, 3)}"

    def __eq__(self, other):
        return isinstance(other, TruthTableRow) and self.formula == other.formula

    def __len__(self):
        return self.n

    def __hash__(self):
        return hash(self.formula)

    def __neg__(self):
        return TruthTableRow(
            -self.formula, self.n, tp=self.fn, fp=self.tn, fn=self.tp, tn=self.fp
        )


class TruthTable:
    def __init__(self, rows: Iterable[TruthTableRow] = ()):
        self.row_hashes = set([])
        self.__rows = []
        for row in rows:
            row_hash = hash(row)
            if row_hash not in self.row_hashes:
                self.row_hashes.add(row_hash)
                self.__rows.append(row)

    def __deepcopy__(self, memodict=None):
        return TruthTable([copy.copy(row) for row in self.__rows])

    def __repr__(self):
        return f"TruthTable({repr(self.__rows)})"

    def __str__(self):
        return "\n".join(map(str, self.__rows))

    def __getitem__(self, item: int | language.Formula) -> TruthTableRow:
        if isinstance(item, int):
            return self.__rows[item]

        assert isinstance(item, language.Formula)

        try:
            return next(row for row in self.__rows if row.formula == item)
        except StopIteration:
            raise KeyError(item)

    def __len__(self):
        return len(self.__rows)

    def __iter__(self) -> Iterable[TruthTableRow]:
        return iter(self.__rows)

    def append(self, row: TruthTableRow):
        row_hash = hash(row)
        if row_hash not in self.row_hashes:
            self.row_hashes.add(row_hash)
            self.__rows.append(row)

    def remove(self, row: TruthTableRow):
        self.__rows.remove(row)
        self.row_hashes.remove(hash(row))

    def __add__(self, other: "TruthTable") -> "TruthTable":
        return TruthTable(self.__rows + other.__rows)

    def __iadd__(self, other: "TruthTable") -> "TruthTable":
        for row in other.__rows:
            self.append(row)

        return self

    def evaluate(
        self,
        inputs: Set[Input],
        graph: gg.GrammarGraph,
        rows_parallel: bool = False,
    ) -> "TruthTable":
        """If lazy is True, then column evaluation stops as soon as result_threshold can no longer be
        reached. E.g., if result_threshold is .9 and there are 100 inputs, then after more than
        10 negative results, 90% positive results is no longer possible."""

        if rows_parallel:
            with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
                self.__rows = set(
                    pool.map(
                        lambda row: row.evaluate(inputs, graph),
                        self.__rows,
                    )
                )
        else:
            for row in self.__rows:
                row.evaluate(inputs, graph)

        return self
