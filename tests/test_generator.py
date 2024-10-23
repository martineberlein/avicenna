import unittest

from isla.language import parse_isla

from avicenna.data import Input
from avicenna.learning.table import Candidate
from avicenna.generator.generator import (
    ISLaSolverGenerator,
    ISLaGrammarBasedGenerator,
    MutationBasedGenerator,
)
from avicenna.generator.engine import ProcessBasedParallelEngine, SingleEngine

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
        constraint1 = """(forall <number> elem in start:
              (<= (str.to.int elem) (str.to.int "-1")) and
        exists <function> elem_0 in start:
              (= elem_0 "sqrt"))
          """
        constraint2 = """
            exists <function> elem in start:
                (= elem "cos")
        """

        constraint3 = """not(forall <number> elem in start:
              (<= (str.to.int elem) (str.to.int "-1")) and
        exists <function> elem_0 in start:
              (= elem_0 "sqrt"))
          """
        constraint1 = parse_isla(constraint1)
        constraint2 = parse_isla(constraint2)
        constraint3 = parse_isla(constraint3)

        candidate1 = Candidate(formula=constraint1)
        candidate2 = Candidate(formula=constraint2)
        candidate3 = Candidate(formula=constraint3)

        generated_inputs = set()
        for con in [candidate1, candidate2, candidate3, candidate1, candidate2, ]:
            generator = ISLaSolverGenerator(self.calculator.get_grammar())
            result = generator.generate_test_inputs(candidate=con)
            if result:
                generated_inputs.update(result)

        for inp in generated_inputs:
            print(inp)
        print(len(generated_inputs))
        self.assertFalse(len(generated_inputs) == 0)

    def test_isla_parallel(self):
        constraint1 = """(forall <number> elem in start:
              (<= (str.to.int elem) (str.to.int "-1")) and
        exists <function> elem_0 in start:
              (= elem_0 "sqrt"))
          """
        constraint2 = """
            exists <function> elem in start:
                (= elem "cos")
        """

        constraint3 = """not(forall <number> elem in start:
              (<= (str.to.int elem) (str.to.int "-1")) and
        exists <function> elem_0 in start:
              (= elem_0 "sqrt"))
          """

        candidate1 = Candidate(formula=parse_isla(constraint1))
        candidate2 = Candidate(formula=parse_isla(constraint2))
        candidate3 = Candidate(formula=parse_isla(constraint3))

        generator = ISLaSolverGenerator(self.calculator.get_grammar(), enable_optimized_z3_queries=False)
        engine = ProcessBasedParallelEngine(generator=generator, workers=2)
        generated_inputs = engine.generate([candidate1, candidate2, candidate3, candidate1, candidate2,])

        for inp in generated_inputs:
            print(inp)
        print(len(generated_inputs))
        self.assertTrue(len(generated_inputs) > 0)


if __name__ == "__main__":
    unittest.main()
