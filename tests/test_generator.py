import unittest

from avicenna.generator import ISLaSolverGenerator, ISLaGrammarBasedGenerator, MutationBasedGenerator
from avicenna_formalizations.calculator import grammar
from avicenna.input import Input

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
        test_inputs = set([Input.from_str(grammar, str_inp) for str_inp in ['\x01 5 hello abc']])
        generator = MutationBasedGenerator(grammar, 10, 10, oracle, test_inputs)

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

if __name__ == '__main__':
    unittest.main()
