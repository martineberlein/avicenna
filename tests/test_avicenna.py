import unittest

from avicenna_formalizations.calculator import oracle, grammar, initial_inputs
from avicenna.avicenna import Avicenna


class TestAvicenna(unittest.TestCase):
    def test_check_initial_passing_failing_requirement(self):
        test_cases = [
            # Each tuple contains: initial_inputs list, expected_exception boolean
            (["sqrt(900)", "cos(10)"], True),
            (["sqrt(-900)", "sqrt(-10)"], True),
            (["sqrt(-900)", "cos(10)"], False),
        ]

        for _initial_inputs, expects_exception in test_cases:
            with self.subTest(initial_inputs=_initial_inputs):
                if expects_exception:
                    self.assertRaises(
                        AssertionError,
                        lambda: Avicenna(grammar=grammar, oracle=oracle, initial_inputs=_initial_inputs),
                    )
                else:
                    # If no exception is expected, attempt instantiation and catch any unexpected exceptions.
                    try:
                        Avicenna(grammar=grammar, oracle=oracle, initial_inputs=_initial_inputs)
                    except AssertionError:
                        self.fail(f"Avicenna raised AssertionError unexpectedly with initial_inputs={_initial_inputs}")

    def test_timeout(self):
        avicenna = Avicenna(grammar=grammar, oracle=oracle, initial_inputs=initial_inputs, timeout_seconds=1)
        self.assertRaises(
            TimeoutError,
            lambda: avicenna.explain(),
        )


if __name__ == "__main__":
    unittest.main()
