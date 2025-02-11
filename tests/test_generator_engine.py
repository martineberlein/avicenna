import unittest

from debugging_benchmark.calculator.calculator import calculator_grammar

from isla import language
from avicenna.generator.engine import SingleEngine, ParallelEngine, ProcessBasedParallelEngine
from avicenna.generator import generator
from avicenna.learning.table import Candidate
from avicenna.data import Input


class TestEngine(unittest.TestCase):

    def setUp(self):
        # Create Candidate instances using actual ISLa formulas
        formula1 = """exists <function> elem_0 in start:
            (= elem_0 "cos")
        """
        formula1 = language.parse_isla(formula1)
        formula2 = """exists <function> elem_0 in start:
            (= elem_0 "sqrt")
        """
        formula2 = language.parse_isla(formula2)

        self.candidate1 = Candidate(formula=formula1)
        self.candidate2 = Candidate(formula=formula2)

        # Create Generator instances
        self.isla_solver_generator = generator.ISLaSolverGenerator(calculator_grammar)
        self.isla_grammar_based_generator = generator.ISLaGrammarBasedGenerator(calculator_grammar)

    def test_single_engine_generate(self):
        engine = SingleEngine(self.isla_solver_generator)
        candidates = [self.candidate1, self.candidate2]
        test_inputs = engine.generate(candidates)
        self.assertTrue(len(test_inputs) > 0)
        # Check that the test_inputs contain expected patterns
        for inp in test_inputs:
            self.assertIsInstance(inp, Input)
        print("SingleEngine test_inputs:", [str(inp) for inp in test_inputs])

    def test_parallel_engine_generate(self):
        engine = ParallelEngine(self.isla_grammar_based_generator, workers=2)
        candidates = [self.candidate1, self.candidate2]
        test_inputs = engine.generate(candidates)
        self.assertTrue(len(test_inputs) > 0)
        # Check that the test_inputs contain expected patterns
        for inp in test_inputs:
            self.assertIsInstance(inp, Input)
        print("ParallelEngine test_inputs:", [str(inp) for inp in test_inputs])

    def test_process_based_engine_generate(self):
        engine = ProcessBasedParallelEngine(self.isla_solver_generator, workers=2)
        candidates = [self.candidate1, self.candidate2]
        test_inputs = engine.generate(candidates)
        self.assertTrue(len(test_inputs) > 0)
        for inp in test_inputs:
            self.assertIsInstance(inp, Input)
        print("ProcessBasedParallelEngine test_inputs:", [str(inp) for inp in test_inputs])

    def test_parallel_engine_incompatible_generator(self):
        # ISLaSolverGenerator is incompatible with ParallelEngine
        with self.assertRaises(ValueError) as context:
            engine = ParallelEngine(self.isla_solver_generator)
        self.assertTrue("not compatible with the ParallelEngine" in str(context.exception))

    def test_parallel_engine_compatible_generator(self):
        # ISLaGrammarBasedGenerator is compatible with ParallelEngine
        engine = ParallelEngine(self.isla_grammar_based_generator, workers=2)
        candidates = [self.candidate1, self.candidate2]
        test_inputs = engine.generate(candidates)
        self.assertTrue(len(test_inputs) > 0)
        for inp in test_inputs:
            self.assertIsInstance(inp, Input)
        print("ParallelEngine with compatible generator test_inputs:", [str(inp) for inp in test_inputs])


if __name__ == '__main__':
    unittest.main()

