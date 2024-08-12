import unittest
import time

from debugging_benchmark.calculator.calculator import calculator_oracle, calculator_grammar, calculator_initial_inputs
from avicenna import Avicenna

from resources.subjects import get_calculator_subject


class TestAvicenna(unittest.TestCase):

    @unittest.skip("Skipping test")
    def test_check_initial_passing_failing_requirement(self):
        test_cases = [
            # Each tuple contains: initial_inputs list, expected_exception boolean
            (["sqrt(900)", "cos(10)"], True),
            #(["sqrt(-900)", "sqrt(-10)"], True),
            (["sqrt(-900)", "cos(10)"], False),
        ]

        for _initial_inputs, expects_exception in test_cases:
            with self.subTest(initial_inputs=_initial_inputs):
                if expects_exception:
                    self.assertRaises(
                        ValueError,
                        lambda: Avicenna(
                            grammar=calculator_grammar,
                            oracle=calculator_oracle,
                            initial_inputs=_initial_inputs,
                        ).explain(),
                    )
                else:
                    # If no exception is expected, attempt instantiation and catch any unexpected exceptions.
                    try:
                        Avicenna(
                            grammar=calculator_grammar,
                            oracle=calculator_oracle,
                            initial_inputs=_initial_inputs,
                        )
                    except AssertionError:
                        self.fail(
                            f"Avicenna raised AssertionError unexpectedly with initial_inputs={_initial_inputs}"
                        )

    import time

    def test_timeout(self):
        calculator = get_calculator_subject()
        avicenna = Avicenna(
            grammar=calculator.get_grammar(),
            oracle=calculator.get_oracle(),
            initial_inputs=calculator.get_initial_inputs(),
            timeout_seconds=4,
        )

        start_time = time.time()
        _ = avicenna.explain()

        duration = time.time() - start_time
        self.assertAlmostEqual(duration, 4, delta=2)  # Check if it ran for ~1 second


if __name__ == "__main__":
    unittest.main()
