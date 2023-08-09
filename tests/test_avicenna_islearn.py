import unittest

from islearn.learner import TruthTable, TruthTableRow
from isla.language import ISLaUnparser
from isla.fuzzer import GrammarFuzzer

from avicenna_formalizations.calculator import (
    grammar,
    oracle,
)
from avicenna_formalizations import get_pattern_file_path
from avicenna.input import Input
from avicenna.oracle import OracleResult
from avicenna.islearn import AviIslearn, AvicennaTruthTable


class TestAvicennaIslearn(unittest.TestCase):

    def setUp(self) -> None:
        inputs = [
            ("sqrt(-901)", OracleResult.BUG),
            ("sqrt(-8)", OracleResult.BUG),
            ("sqrt(10)", OracleResult.NO_BUG),
            ("cos(1)", OracleResult.NO_BUG),
            ("sin(99)", OracleResult.NO_BUG),
            ("tan(-20)", OracleResult.NO_BUG),
            ("sqrt(-20)", OracleResult.BUG),
        ]
        self.test_inputs = set(
            [Input.from_str(grammar, inp, oracle) for inp, oracle in inputs]
        )

    def test_avicenna_islearn(self):
        avi_islearn = AviIslearn(grammar, oracle, pattern_file=str(get_pattern_file_path()))
        exclude_nonterminals = ["<digits>", "<maybe_digits>", "<onenine>", "<arith_expr>", "<start>", "<digit>"]

        precision_truth_table = AvicennaTruthTable()
        recall_truth_table = AvicennaTruthTable()
        result, precision_truth_table , recall_truth_table = avi_islearn.learn_failure_invariants(self.test_inputs, self.test_inputs, precision_truth_table, recall_truth_table, exclude_nonterminals)

        # for formula in result.keys():
            # print(ISLaUnparser(formula).unparse())
        # self.assertEqual(True, False)  # add assertion here

        inputs = [
            ("sqrt(-20)", OracleResult.BUG),
            ("sqrt(-3)", OracleResult.BUG),
            ("sqrt(112)", OracleResult.NO_BUG),
            ("cos(14)", OracleResult.NO_BUG),
            ("sin(9123)", OracleResult.NO_BUG),
            ("tan(-2)", OracleResult.NO_BUG),
        ]
        new_inputs = set(
            [Input.from_str(grammar, inp, oracle) for inp, oracle in inputs]
        )
        # self.test_inputs.update(new_inputs)
        new_inputs = new_inputs.difference(self.test_inputs)

        #precision_truth_table = AvicennaTruthTable()
        #recall_truth_table = AvicennaTruthTable()
        result, _ , _ = avi_islearn.learn_failure_invariants(new_inputs, self.test_inputs, precision_truth_table, recall_truth_table, exclude_nonterminals)

        failure_constraints = list(
            map(lambda p: (p[1], ISLaUnparser(p[0]).unparse()), result.items())
        )
        for f in failure_constraints:
            print(f)

    def test_iterative_addition(self):
        fuzzer = GrammarFuzzer(grammar)
        test_inputs = set()
        for _ in range(200):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(tree=inp, oracle=oracle(str(inp))))

        avi_islearn = AviIslearn(grammar, oracle, pattern_file=str(get_pattern_file_path()))
        exclude_nonterminals = ["<digits>", "<maybe_digits>", "<onenine>", "<arith_expr>", "<start>", "<digit>"]

        precision_truth_table = AvicennaTruthTable()
        recall_truth_table = AvicennaTruthTable()
        result, precision_truth_table , recall_truth_table = avi_islearn.learn_failure_invariants(test_inputs, test_inputs, precision_truth_table, recall_truth_table, exclude_nonterminals)

        inputs = [
            ("sqrt(-20)", OracleResult.BUG),
            ("sqrt(-3)", OracleResult.BUG),
            ("sqrt(112)", OracleResult.NO_BUG),
            ("cos(14)", OracleResult.NO_BUG),
            ("sin(9123)", OracleResult.NO_BUG),
            ("tan(-2)", OracleResult.NO_BUG),
        ]
        new_inputs = set(
            [Input.from_str(grammar, inp, oracle) for inp, oracle in inputs]
        )
        #test_inputs.update(new_inputs)
        new_inputs = new_inputs.difference(test_inputs)
        #new_inputs = test_inputs.union(new_inputs)

        #precision_truth_table = AvicennaTruthTable()
        #recall_truth_table = AvicennaTruthTable()
        result, _ , _ = avi_islearn.learn_failure_invariants(new_inputs, test_inputs, precision_truth_table, recall_truth_table, exclude_nonterminals)

        failure_constraints = list(
            map(lambda p: (p[1], ISLaUnparser(p[0]).unparse()), result.items())
        )
        for f in failure_constraints:
            print(f)

    def test_old(self):
        from avicenna.islearn import AvicennaISlearn
        fuzzer = GrammarFuzzer(grammar)
        test_inputs = set()
        for _ in range(200):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(tree=inp, oracle=oracle(str(inp))))

        def map_to_bool(result: OracleResult) -> bool:
            match result:
                case OracleResult.BUG:
                    return True
                case OracleResult.NO_BUG:
                    return False
                case _:
                    return False
        def boolean_oracle(inp_):
            return map_to_bool(oracle(inp_))

        boolean_oracle = boolean_oracle
        islearn: AvicennaISlearn = AvicennaISlearn(
            grammar=grammar,
            prop=boolean_oracle,
            reduce_inputs_for_learning=False,
            reduce_all_inputs=False,
            filter_inputs_for_learning_by_kpaths=False,
            do_generate_more_inputs=False,
            generate_new_learning_samples=False,
            pattern_file=str(get_pattern_file_path()))

        exclude_nonterminals = ["<digits>", "<maybe_digits>", "<onenine>", "<arith_expr>", "<start>", "<digit>"]
        new_candidates, precision_truth_table, recall_truth_table = islearn.learn_failure_invariants(
            test_inputs, exclude_nonterminals
        )
        new_candidates, precision_truth_table, recall_truth_table = islearn.learn_failure_invariants(
            test_inputs, exclude_nonterminals
        )

        failure_constraints = list(
            map(lambda p: (p[1], ISLaUnparser(p[0]).unparse()), new_candidates.items())
        )
        for f in failure_constraints:
            print(f)

    def test_sets(self):
        old = {1, 2, 3, 4, 5}
        new = {4, 5, 6, 7, 8}

        print(new.difference(old))  # Output: {1, 2, 3}


if __name__ == '__main__':
    unittest.main()
