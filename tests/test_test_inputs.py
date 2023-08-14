import unittest
from typing import Tuple

from isla.derivation_tree import DerivationTree
from fuzzingbook.Parser import EarleyParser, is_valid_grammar, Grammar

from avicenna_formalizations.calculator import grammar, oracle
from avicenna.oracle import OracleResult
from avicenna.input import Input


class TestInputs(unittest.TestCase):
    def setUp(self) -> None:
        inputs = {"sqrt(-900)", "cos(10)"}

        self.test_inputs = set()
        for inp in inputs:
            self.test_inputs.add(
                Input(
                    DerivationTree.from_parse_tree(
                        next(EarleyParser(grammar).parse(inp))
                    )
                )
            )

    def test_test_inputs(self):
        inputs = {"sqrt(-900)", "cos(10)"}
        oracles = [OracleResult.BUG, OracleResult.NO_BUG]
        test_inputs = set()
        for inp, oracle in zip(inputs, oracles):
            test_input = Input(
                DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
            )
            test_input.oracle = oracle
            test_inputs.add(test_input)

        self.assertEqual(inputs, set(map(lambda f: str(f.tree), test_inputs)))

        for inp, orc in zip(inputs, oracles):
            self.assertIn(
                (inp, orc), set(map(lambda x: (str(x.tree), x.oracle), test_inputs))
            )

        self.assertFalse(
            set(map(lambda f: str(f.tree), test_inputs)).__contains__("cos(X)")
        )

    def test_input_execution(self):
        for inp in self.test_inputs:
            inp.oracle = oracle(inp)

    def test_hash(self):
        grammar: Grammar = {
            "<start>": ["<number>"],
            "<number>": ["<maybe_minus><one_nine>"],
            "<maybe_minus>": ["-", ""],
            "<one_nine>": [str(i) for i in range(1, 10)],
        }
        assert is_valid_grammar(grammar=grammar)

        initial_test_inputs = ["-8", "-8"]

        test_inputs = set()
        for inp in initial_test_inputs:
            test_inputs.add(
                Input(
                    DerivationTree.from_parse_tree(
                        next(EarleyParser(grammar).parse(inp))
                    )
                )
            )

        self.assertEqual(1, len(test_inputs))


if __name__ == "__main__":
    unittest.main()
