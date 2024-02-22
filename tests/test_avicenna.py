import unittest

from avicenna_formalizations.calculator import oracle, grammar
from avicenna.avicenna import Avicenna


class TestAvicenna(unittest.TestCase):
    def test_check_initial_passing_failing_requirement(self):
        test_cases = [
            # Each tuple contains: initial_inputs list, expected_exception boolean
            (["sqrt(900)", "cos(10)"], True),
            (["sqrt(-900)", "sqrt(-10)"], True),
            (["sqrt(-900)", "cos(10)"], False),
        ]

        for initial_inputs, expects_exception in test_cases:
            with self.subTest(initial_inputs=initial_inputs):
                if expects_exception:
                    self.assertRaises(
                        AssertionError,
                        lambda: Avicenna(grammar=grammar, oracle=oracle, initial_inputs=initial_inputs),
                    )
                else:
                    # If no exception is expected, attempt instantiation and catch any unexpected exceptions.
                    try:
                        Avicenna(grammar=grammar, oracle=oracle, initial_inputs=initial_inputs)
                    except AssertionError:
                        self.fail(f"Avicenna raised AssertionError unexpectedly with initial_inputs={initial_inputs}")


if __name__ == "__main__":
    unittest.main()
