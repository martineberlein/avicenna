import unittest

from isla.language import ISLaUnparser
from isla.fuzzer import GrammarFuzzer

from debugging_framework.input.oracle import OracleResult

from avicenna_formalizations.calculator import (
    grammar,
    oracle,
)
from avicenna_formalizations import get_pattern_file_path
from avicenna.input import Input
from avicenna.pattern_learner import AviIslearn, AvicennaTruthTable


class TestAvicennaIslearn(unittest.TestCase):
    def setUp(self) -> None:
        inputs = [
            ("sqrt(-901)", OracleResult.FAILING),
            ("sqrt(-8)", OracleResult.FAILING),
            ("sqrt(10)", OracleResult.PASSING),
            ("cos(1)", OracleResult.PASSING),
            ("sin(99)", OracleResult.PASSING),
            ("tan(-20)", OracleResult.PASSING),
            ("sqrt(-20)", OracleResult.FAILING),
        ]
        self.test_inputs = set(
            [Input.from_str(grammar, inp, inp_oracle) for inp, inp_oracle in inputs]
        )

    def test_avicenna_islearn(self):
        print(str(get_pattern_file_path()))
        avi_islearn = AviIslearn(grammar, pattern_file=str(get_pattern_file_path()))
        exclude_nonterminals = [
            "<digits>",
            "<maybe_digits>",
            "<onenine>",
            "<arith_expr>",
            "<start>",
            "<digit>",
        ]

        precision_truth_table = AvicennaTruthTable()
        recall_truth_table = AvicennaTruthTable()
        result = avi_islearn.learn_failure_invariants(
            self.test_inputs,
            precision_truth_table,
            recall_truth_table,
            exclude_nonterminals,
        )

        inputs = [
            ("sqrt(-20)", OracleResult.FAILING),
            ("sqrt(-3)", OracleResult.FAILING),
            ("sqrt(112)", OracleResult.PASSING),
            ("cos(14)", OracleResult.PASSING),
            ("sin(9123)", OracleResult.PASSING),
            ("tan(-2)", OracleResult.PASSING),
        ]
        new_inputs = set(
            [Input.from_str(grammar, inp, inp_oracle) for inp, inp_oracle in inputs]
        )
        # self.test_inputs.update(new_inputs)
        new_inputs = new_inputs.difference(self.test_inputs)

        # precision_truth_table = AvicennaTruthTable()
        # recall_truth_table = AvicennaTruthTable()
        result = avi_islearn.learn_failure_invariants(
            new_inputs, precision_truth_table, recall_truth_table, exclude_nonterminals
        )

        failure_constraints = list(
            map(lambda p: ((p[1], p[2]), ISLaUnparser(p[0]).unparse()), result)
        )
        for f in failure_constraints:
            print(f)

    def test_iterative_addition(self):
        print(str(get_pattern_file_path()))
        fuzzer = GrammarFuzzer(grammar)
        test_inputs = set()
        for _ in range(200):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(inp, oracle(str(inp))))

        avi_islearn = AviIslearn(grammar, pattern_file=str(get_pattern_file_path()))
        exclude_nonterminals = [
            "<digits>",
            "<maybe_digits>",
            "<onenine>",
            "<arith_expr>",
            "<start>",
            "<digit>",
        ]

        precision_truth_table = AvicennaTruthTable()
        recall_truth_table = AvicennaTruthTable()
        result = avi_islearn.learn_failure_invariants(
            test_inputs, precision_truth_table, recall_truth_table, exclude_nonterminals
        )

        inputs = [
            ("sqrt(-20)", OracleResult.FAILING),
            ("sqrt(-3)", OracleResult.FAILING),
            ("sqrt(112)", OracleResult.PASSING),
            ("cos(14)", OracleResult.PASSING),
            ("sin(9123)", OracleResult.PASSING),
            ("tan(-2)", OracleResult.PASSING),
        ]
        new_inputs = set(
            [Input.from_str(grammar, inp, inp_oracle) for inp, inp_oracle in inputs]
        )
        # test_inputs.update(new_inputs)
        new_inputs = new_inputs.difference(test_inputs)
        # new_inputs = test_inputs.union(new_inputs)

        # precision_truth_table = AvicennaTruthTable()
        # recall_truth_table = AvicennaTruthTable()
        result = avi_islearn.learn_failure_invariants(
            new_inputs, precision_truth_table, recall_truth_table, exclude_nonterminals
        )

        failure_constraints = list(
            map(lambda p: ((p[1], p[2]), ISLaUnparser(p[0]).unparse()), result)
        )
        for f in failure_constraints:
            print(f)


if __name__ == "__main__":
    unittest.main()
