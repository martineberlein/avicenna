import unittest

import isla.language as language
import isla.parser
from grammar_graph import gg
from debugging_framework.input.oracle import OracleResult

from avicenna.data import Input
from avicenna.learning.table import Candidate, CandidateSet
from resources.subjects import get_calculator_subject


class TestTables(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.calculator = get_calculator_subject()
        cls.test_inputs = set(
            [
                Input.from_str(cls.calculator.get_grammar(), inp, inp_oracle)
                for inp, inp_oracle in [
                    ("sqrt(-901)", OracleResult.FAILING),
                    ("sqrt(-8)", OracleResult.FAILING),
                    ("sqrt(10)", OracleResult.PASSING),
                    ("cos(1)", OracleResult.PASSING),
                    ("sin(99)", OracleResult.PASSING),
                    ("tan(-20)", OracleResult.PASSING),
                    ("sqrt(-20)", OracleResult.FAILING),
                ]
            ]
        )
        cls.graph = gg.GrammarGraph.from_grammar(cls.calculator.get_grammar())
        formula1 = """forall <number> elem in start:
            (<= (str.to.int elem) (str.to.int "-1"))
        """
        cls.formula1 = isla.language.parse_isla(formula1)
        formula2 = """exists <function> elem_0 in start:
            (= elem_0 "sqrt")
        """
        cls.formula2 = isla.language.parse_isla(formula2)

    def test_candidates(self):
        """Test the Candidate class."""
        candidate = Candidate(formula=self.formula1)
        print(candidate)
        self.assertIsInstance(candidate, Candidate)
        self.assertIsInstance(candidate.formula, language.Formula)
        candidate.evaluate(self.test_inputs, self.graph)

        self.assertEqual(len(candidate.inputs), 7)
        self.assertEqual(len(candidate.failing_inputs_eval_results), 3)
        self.assertEqual(len(candidate.passing_inputs_eval_results), 4)

        self.assertEqual(candidate.recall(), 1.0)
        self.assertEqual(candidate.specificity(), 0.75)
        self.assertEqual(candidate.precision(), 0.75)

    def test_candidate_with_more_inputs(self):
        candidate = Candidate(formula=self.formula1)
        candidate.evaluate(self.test_inputs, self.graph)
        new_inputs = set([
            Input.from_str(self.calculator.get_grammar(), inp, inp_oracle)
            for inp, inp_oracle in [
                ("sqrt(-2)", OracleResult.FAILING),
                ("sin(-23)", OracleResult.PASSING),
                ("sin(-10)", OracleResult.PASSING),
            ]
        ])
        candidate.evaluate(new_inputs, self.graph)
        self.assertEqual(candidate.recall(), 1.0)
        self.assertEqual(candidate.specificity(), 0.5)
        self.assertAlmostEqual(candidate.precision(), 0.571, places=3)

    def test_candidate_set(self):
        """Test the CandidateSet class."""
        candidate1 = Candidate(formula=self.formula1)
        candidate2 = Candidate(formula=self.formula2)

        candidate_set = CandidateSet()
        candidate_set.append(candidate1)
        self.assertNotIn(candidate2, candidate_set)

        candidate_set.append(candidate2)
        candidate_set.append(candidate1)

        self.assertEqual(len(candidate_set), 2)
        for candidate in [candidate1, candidate2]:
            candidate_set.remove(candidate)

        self.assertEqual(len(candidate_set), 0)

    def test_candidate_combination(self):
        candidate1 = Candidate(formula=self.formula1)
        candidate2 = Candidate(formula=self.formula2)

        candidate_set = CandidateSet([candidate1, candidate2])
        for candidate in candidate_set:
            candidate.evaluate(self.test_inputs, self.graph)

        conjunction = candidate1 & candidate2
        self.assertIsInstance(conjunction, Candidate)
        self.assertEqual(conjunction.recall(), 1.0)
        self.assertEqual(conjunction.specificity(), 1.0)
        self.assertEqual(conjunction.precision(), 1.0)
        self.assertEqual(len(conjunction.inputs), 7)
        self.assertEqual(len(conjunction.failing_inputs_eval_results), 3)
        self.assertEqual(len(conjunction.passing_inputs_eval_results), 4)


if __name__ == '__main__':
    unittest.main()
