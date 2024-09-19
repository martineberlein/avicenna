import unittest
from typing import List

import isla.language as language
from debugging_framework.input.oracle import OracleResult

from avicenna.data import Input
from avicenna.learning.exhaustive import ExhaustivePatternCandidateLearner
from avicenna.learning.table import Candidate

from resources.subjects import get_calculator_subject


class TestExhaustivePatternLearner(unittest.TestCase):

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
        cls.exclude_nonterminals = [
            "<digits>",
            "<maybe_digits>",
            "<onenine>",
            "<arith_expr>",
            "<start>",
            "<digit>",
        ]
        cls.exhaustive_learner = ExhaustivePatternCandidateLearner(cls.calculator.get_grammar())

    def verify_candidates(self, result: List[Candidate], expected_length: int | None):
        if expected_length:
            self.assertEqual(len(result), expected_length)
        if len(result) != expected_length:
            print("FAILED, expected ", expected_length, " but got ", len(result))
        for candidate in result:
            print(language.ISLaUnparser(candidate.formula).unparse())
            print("Specificity: ", candidate.specificity(), " Recall: ", candidate.recall(), " Precision: ", candidate.precision(), "length: ", len(candidate.inputs))
            self.assertIsInstance(candidate.formula, language.Formula)
            self.assertIsInstance(candidate.precision(), float)
            self.assertIsInstance(candidate.recall(), float)

    def test_exhaustive_pattern_learner(self):
        """Test the exhaustive pattern learner with initial inputs."""
        result = self.exhaustive_learner.learn_candidates(
            self.test_inputs, self.exclude_nonterminals
        )
        self.verify_candidates(result, expected_length=58)
        self.exhaustive_learner.reset()

    def test_reset(self):
        """Test the reset method."""
        self.exhaustive_learner.learn_candidates(
            self.test_inputs, self.exclude_nonterminals
        )
        self.exhaustive_learner.reset()
        self.assertEqual(len(self.exhaustive_learner.candidates), 0)
        self.assertEqual(len(self.exhaustive_learner.all_positive_inputs), 0)
        self.assertEqual(len(self.exhaustive_learner.all_negative_inputs), 0)

    def test_exhaustive_pattern_learner_more_runs(self):
        """Test the exhaustive pattern learner with additional inputs."""
        result = self.exhaustive_learner.learn_candidates(
            self.test_inputs, self.exclude_nonterminals
        )
        self.verify_candidates(result, expected_length=58)

        more_inputs = set(
            [
                Input.from_str(self.calculator.get_grammar(), inp, inp_oracle)
                for inp, inp_oracle in [
                    ("sqrt(-1)", OracleResult.FAILING),
                    ("sqrt(-3)", OracleResult.FAILING),
                    ("sqrt(1)", OracleResult.PASSING),
                    ("cos(14)", OracleResult.PASSING),
                    ("sin(9123)", OracleResult.PASSING),
                    ("tan(-2)", OracleResult.PASSING),
                ]
            ]
        )
        result = self.exhaustive_learner.learn_candidates(
            more_inputs, self.exclude_nonterminals
        )
        self.verify_candidates(result, expected_length=33)
        best_candidates = self.exhaustive_learner.get_best_candidates()
        self.verify_candidates(best_candidates, expected_length=5)
        self.exhaustive_learner.reset()

    def test_get_candidates(self):
        """Test the get_candidates method."""
        _ = self.exhaustive_learner.learn_candidates(
            self.test_inputs, self.exclude_nonterminals
        )
        result = self.exhaustive_learner.get_candidates()
        self.verify_candidates(result, expected_length=58)
        self.exhaustive_learner.reset()

    def test_get_best_candidates(self):
        """Test the get_best_candidates method."""
        _ = self.exhaustive_learner.learn_candidates(
            self.test_inputs, self.exclude_nonterminals
        )
        result = self.exhaustive_learner.get_best_candidates()
        self.verify_candidates(result, expected_length=14)
        self.exhaustive_learner.reset()


if __name__ == "__main__":
    unittest.main()
