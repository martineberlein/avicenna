import unittest
from typing import List

from debugging_framework.fuzzingbook.grammar import is_valid_grammar, Grammar

from avicenna.data import Input, OracleResult
from resources.subjects import get_calculator_subject


class TestInputs(unittest.TestCase):
    def setUp(self) -> None:
        inputs = {"sqrt(-900)", "cos(10)"}

        self.calculator = get_calculator_subject()
        self.test_inputs = {
            Input.from_str(self.calculator.grammar, inp) for inp in inputs
        }

    def test_parsed_inputs_have_expected_trees_and_oracles(self):
        inputs = {
            ("sqrt(-900)", OracleResult.FAILING),
            ("cos(10)", OracleResult.PASSING),
        }
        labeled_inputs = {
            inp.update_oracle(self.calculator.oracle(inp)) for inp in self.test_inputs
        }

        # Validate oracles and exceptions
        for inp in labeled_inputs:
            oracle, exception = inp.oracle
            self.assertIsInstance(oracle, OracleResult)
            self.assertIsInstance(
                exception, ValueError if oracle == OracleResult.FAILING else type(None)
            )

        # Check if input strings are as expected
        input_strings = {str(inp) for inp in labeled_inputs}
        expected_strings = {x[0] for x in inputs}
        self.assertSetEqual(input_strings, expected_strings)

        # Validate actual trees and oracles
        actual_trees = {(str(inp), inp.oracle[0]) for inp in labeled_inputs}
        self.assertSetEqual(actual_trees, inputs)

        # Additional specific assertion
        self.assertNotIn("cos(X)", {str(f.tree) for f in labeled_inputs})

    def test_input_execution_is_oracle_result(self):
        parsed_inputs = {
            inp.update_oracle(self.calculator.oracle(inp)) for inp in self.test_inputs
        }

        for inp in parsed_inputs:
            oracle, exception = inp.oracle
            self.assertIsInstance(oracle, OracleResult)

    def test_input_from_str(self):
        input_strings = ["sqrt(-900)"]
        expected_oracle_result = [OracleResult.FAILING]

        parsed_input: List[Input] = [
            Input.from_str(self.calculator.grammar, inp, self.calculator.oracle(inp))
            for inp in input_strings
        ]
        for inp, expected, expected_oracle in zip(
            parsed_input, input_strings, expected_oracle_result
        ):
            self.assertIsInstance(inp, Input)
            self.assertEqual(str(inp.tree), expected)
            oracle, exception = inp.oracle
            self.assertEqual(oracle, expected_oracle)

    @unittest.skip
    def test_input_immutable(self):
        inp = next(iter(self.test_inputs))
        original_tree = inp.tree
        original_oracle = inp.oracle

        with self.assertRaises(AttributeError):
            inp.tree = "new tree"

        with self.assertRaises(AttributeError):
            inp.oracle = OracleResult.PASSING

        self.assertEqual(inp.tree, original_tree)
        self.assertEqual(inp.oracle, original_oracle)

    def test_invalid_input_string(self):
        invalid_input_string = "invalid_input"

        with self.assertRaises(SyntaxError):
            Input.from_str(self.calculator.grammar, invalid_input_string)

    def test_input_equality(self):
        inp1 = Input.from_str(self.calculator.grammar, "sqrt(-900)")
        inp2 = Input.from_str(self.calculator.grammar, "sqrt(-900)")
        inp3 = Input.from_str(self.calculator.grammar, "cos(10)")

        self.assertEqual(inp1, inp2)
        self.assertNotEqual(inp1, inp3)

    def test_hash(self):
        grammar_simple: Grammar = {
            "<start>": ["<number>"],
            "<number>": ["<maybe_minus><one_nine>"],
            "<maybe_minus>": ["-", ""],
            "<one_nine>": [str(i) for i in range(1, 10)],
        }
        assert is_valid_grammar(grammar=grammar_simple)

        initial_test_inputs = ["-8", "-8"]
        parsed_input: List[Input] = [
            Input.from_str(grammar_simple, inp) for inp in initial_test_inputs
        ]

        self.assertTrue(all(hash(parsed_input[0]) == hash(inp) for inp in parsed_input))
        set_parsed_input = set(parsed_input)
        self.assertEqual(1, len(set_parsed_input))

    def test_input_oracle_is_failing(self):
        for inp in self.test_inputs:
            inp.oracle = self.calculator.oracle(inp)
            oracle, exception = inp.oracle
            if oracle == OracleResult.FAILING:
                self.assertTrue(oracle.is_failing())
            else:
                self.assertFalse(oracle.is_failing())


if __name__ == "__main__":
    unittest.main()
