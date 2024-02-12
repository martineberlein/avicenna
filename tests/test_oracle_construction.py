import unittest
from avicenna.oracle_construction import (
    construct_oracle,
    OracleResult,
    Input,
    UnexpectedResultError,
)
from fuzzingbook.Grammars import Grammar
import string

grammar: Grammar = {
    "<start>": ["<input>"],
    "<input>": ["<first> <second>"],
    "<first>": ["<integer>"],
    "<second>": ["<integer>"],
    "<integer>": ["<one_nine><maybe_digits>"],  # no 0 at the moment
    "<one_nine>": [str(num) for num in range(1, 10)],
    "<digit>": list(string.digits),
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
}


class TestConstructOracle(unittest.TestCase):
    def setUp(self):
        self.error_definitions = {
            UnexpectedResultError: OracleResult.FAILING,
            TimeoutError: OracleResult.UNDEFINED,
        }

    def test_same_result(self):
        def oracle(x, y):
            return x + y

        def under_test(x, y):
            return x + y

        my_oracle = construct_oracle(under_test, oracle, self.error_definitions)
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.PASSING
        )

    def test_different_result(self):
        def oracle(x, y):
            return x + y

        def under_test(x, y):
            return x - y

        my_oracle = construct_oracle(under_test, oracle, self.error_definitions)
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.FAILING
        )

    def test_defined_exception(self):
        def oracle(x, y):
            return x + y

        def under_test(x, y):
            raise TimeoutError()

        my_oracle = construct_oracle(oracle, under_test, self.error_definitions)
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.UNDEFINED
        )

    def test_undefined_exception(self):
        def oracle(x, y):
            return x + y

        def under_test(x, y):
            raise ValueError()

        my_oracle = construct_oracle(oracle, under_test, self.error_definitions)
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.UNDEFINED
        )

    def test_timeout_sleep(self):
        def oracle(x, y):
            return x + y

        def under_test(x, y):
            import time

            time.sleep(2)
            return x + y

        my_oracle = construct_oracle(
            under_test,
            oracle,
            {
                UnexpectedResultError: OracleResult.PASSING,
                TimeoutError: OracleResult.FAILING,
            },
            timeout=1,
        )
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.FAILING
        )

    def test_no_error_definition(self):
        def oracle(x, y):
            return x + y

        def under_test(x, y):
            raise ValueError

        def under_test_unexpected_result_error(x, y):
            raise x + y + 1

        def under_test_timeout(x, y):
            import time

            time.sleep(2)
            return x + y

        my_oracle = construct_oracle(under_test, oracle)
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.FAILING
        )
        my_oracle = construct_oracle(under_test_unexpected_result_error, oracle)
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.FAILING
        )
        my_oracle = construct_oracle(under_test_timeout, oracle)
        self.assertEqual(
            my_oracle(Input.from_str(grammar, "1 1")), OracleResult.FAILING
        )


if __name__ == "__main__":
    unittest.main()
