import unittest

from avicenna.data import Input
from avicenna.generator.generator import (
    ISLaSolverGenerator,
    ISLaGrammarBasedGenerator,
    MutationBasedGenerator,
)

from resources.subjects import get_calculator_subject, get_heartbleed_subject


class TestInputGenerator(unittest.TestCase):

    def setUp(self):
        self.calculator = get_calculator_subject()
        self.heartbleed = get_heartbleed_subject()

    def test_isla_grammar_fuzzer(self):
        generator = ISLaGrammarBasedGenerator(self.calculator.get_grammar())

        generated_inputs = []
        for _ in range(10):
            result = generator.generate()
            generated_inputs.append(result)

        self.assertEqual(len(generated_inputs), 10)
        self.assertTrue(all([isinstance(inp, Input) for inp in generated_inputs]))

    def test_mutation_generator(self):
        test_inputs = set(
            [Input.from_str(self.heartbleed.get_grammar(), str_inp) for str_inp in ["\x01 5 hello abc"]]
        )
        generator = MutationBasedGenerator(
            self.heartbleed.get_grammar(),
            self.heartbleed.get_oracle(),
            test_inputs
        )

        generated_inputs = []
        for _ in range(100):
            result = generator.generate()
            if result:
                generated_inputs.append(result)
            else:
                break

        print(len(generated_inputs))

    def test_isla_solver(self):
        constraint = """(forall <number> elem in start:
              (<= (str.to.int elem) (str.to.int "-1")) and
        exists <function> elem_0 in start:
              (= elem_0 "sqrt"))
          """
        constraint_2 = """
            exists <function> elem in start:
                (= elem "cos")
        """

        constraint_3 = """not(forall <number> elem in start:
              (<= (str.to.int elem) (str.to.int "-1")) and
        exists <function> elem_0 in start:
              (= elem_0 "sqrt"))
          """

        failed = False
        generated_inputs = []
        for con in [constraint, constraint_2, constraint_3]:
            generator = ISLaSolverGenerator(self.calculator.get_grammar(), constraint=con)

            for _ in range(100):
                result = generator.generate()
                if result:
                    generated_inputs.append(result)
                else:
                    break
            failed = len(generated_inputs) == 0 or failed

        print(len(generated_inputs))
        self.assertFalse(failed)


if __name__ == "__main__":
    unittest.main()
