import unittest
from typing import Set, List

from fuzzingbook.Parser import is_valid_grammar, Grammar
from debugging_framework.oracle import OracleResult


from avicenna_formalizations.calculator import grammar, oracle
from avicenna.input import Input
from avicenna.monads import Exceptional

class TestInputs(unittest.TestCase):
    def setUp(self) -> None:
        inputs = {"sqrt(-900)", "cos(10)"}

        self.test_inputs = (
            Exceptional.of(lambda: inputs)
            .map(lambda x: {Input.from_str(grammar, inp_) for inp_ in x})
            .reraise()
            .get()
        )

    def test_parsed_inputs_have_expected_trees_and_oracles(self):
        inputs = {("sqrt(-900)", OracleResult.FAILING),
                  ("cos(10)", OracleResult.PASSING)}

        parsed_inputs = (
            Exceptional.of(lambda : self.test_inputs)
            .map(lambda x: {inp_.update_oracle(oracle(inp_)) for inp_ in x})
            .reraise()
            .get()
        )
        actual_trees = {(str(f.tree), f.oracle )for f in parsed_inputs}

        self.assertEqual(actual_trees, inputs)
        self.assertNotIn("cos(X)", set(map(lambda f: str(f.tree), parsed_inputs)))

    def test_input_execution_is_oracle_result(self):
        parsed_inputs = (
            Exceptional.of(lambda : self.test_inputs)
            .map(lambda x: {inp_.update_oracle(oracle(inp_)) for inp_ in x})
            .reraise()
            .get()
        )

        for inp in parsed_inputs:
            inp.oracle = oracle(inp)
            self.assertIsInstance(inp.oracle, OracleResult)

    def test_input_from_str(self):
        input_strings = ["sqrt(-900)"]
        expected_oracle_result = [OracleResult.FAILING]

        parsed_input: List[Input] = (
            Exceptional.of(lambda : input_strings)
            .map(lambda x: [Input.from_str(grammar, inp_, oracle(inp_)) for inp_ in x])
            .reraise()
            .get()
        )
        for inp, expected, expected_oracle in zip(parsed_input, input_strings, expected_oracle_result):
            self.assertIsInstance(inp, Input)
            self.assertEqual(str(inp.tree), expected)
            self.assertEqual(inp.oracle, expected_oracle)

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
            Input.from_str(grammar, invalid_input_string)

    def test_input_equality(self):
        inp1 = Input.from_str(grammar, "sqrt(-900)")
        inp2 = Input.from_str(grammar, "sqrt(-900)")
        inp3 = Input.from_str(grammar, "cos(10)")

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
        parsed_input: Set[Input] = (
            Exceptional.of(lambda : initial_test_inputs)
            .map(lambda x: {Input.from_str(grammar_simple, inp_) for inp_ in x})
            .reraise()
            .get()
        )

        self.assertEqual(1, len(parsed_input))


if __name__ == "__main__":
    unittest.main()
