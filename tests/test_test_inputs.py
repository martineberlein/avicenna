import unittest
from typing import Tuple

from isla.derivation_tree import DerivationTree
from fuzzingbook.Parser import EarleyParser

from avicenna_formalizations.calculator import grammar_alhazen as grammar, prop
from avicenna.oracle import OracleResult
from avicenna.feature_collector import Collector
from avicenna.features import EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE
from avicenna.input import Input


FEATURES = {EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE}


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

        self.collector = Collector(grammar=grammar, features=FEATURES)

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
            inp.oracle = prop(inp.tree)

    def test_feature_extraction(self):
        self._all_features = self.collector.get_all_features()

        for inp in self.test_inputs:
            inp.features = self.collector.collect_features(inp)


if __name__ == "__main__":
    unittest.main()
