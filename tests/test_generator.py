import unittest

from avicenna.generator.generator import (
    ISLaSolverGenerator,
    ISLaGrammarBasedGenerator,
    MutationBasedGenerator,
)
from avicenna_formalizations.calculator import grammar
from avicenna.input.input import Input
from avicenna.monads import Just, Nothing


class TestInputGenerator(unittest.TestCase):
    def test_isla_grammar_fuzzer(self):
        generator = ISLaGrammarBasedGenerator(grammar)

        generated_inputs = []
        for _ in range(10):
            result = generator.generate()
            if result.is_just():
                generated_inputs.append(result.value())
            else:
                break

        self.assertEqual(len(generated_inputs), 10)

    def test_mutation_generator(self):
        from avicenna_formalizations.heartbeat import grammar, oracle

        test_inputs = set(
            [Input.from_str(grammar, str_inp) for str_inp in ["\x01 5 hello abc"]]
        )
        generator = MutationBasedGenerator(grammar, oracle, test_inputs)

        generated_inputs = []
        for _ in range(100):
            result = generator.generate()
            if result.is_just():
                print(result.value())
                generated_inputs.append(result.value())
            else:
                print("End")

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
            generator = ISLaSolverGenerator(grammar, constraint=con)

            for _ in range(5):
                result = generator.generate()
                if result.is_just():
                    generated_inputs.append(result.value())
                else:
                    break
            failed = len(generated_inputs) == 0 or failed

        self.assertFalse(failed)

    def test_generate_with_monads(self):
        constraint = """(forall <number> elem in start:
                      (<= (str.to.int elem) (str.to.int "-1")) and
                exists <function> elem_0 in start:
                      (= elem_0 "sqrt"))
                  """
        constraint_2 = """
                    exists <function> elem in start:
                        (= elem "cos")
                """
        result = Just({constraint, constraint_2})

        inputs = result.bind(self.generate_inputs).value()
        self.assertEqual(len(inputs), 2)

    @staticmethod
    def generate_inputs(candidate_set):
        generated_inputs = set()
        for _ in candidate_set:
            generator = ISLaGrammarBasedGenerator(grammar)
            for _ in range(1):
                result_ = generator.generate()
                if result_.is_just():
                    generated_inputs.add(result_.value())
                else:
                    break
        if generated_inputs:
            return Just(generated_inputs)
        return Nothing()


if __name__ == "__main__":
    unittest.main()
