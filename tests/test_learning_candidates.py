import unittest

from avicenna.learning.metric import (
    PrecisionFitness,
    RecallFitness,
    RecallPriorityFitness,
    F1ScoreFitness,
    RecallPriorityLengthFitness,
)
from debugging_framework.input.oracle import OracleResult
from avicenna.data import Input
from avicenna.learning.exhaustive import ExhaustivePatternCandidateLearner

from resources.subjects import get_calculator_subject


class TestCandidates(unittest.TestCase):
    """Test the Candidate"""

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
        cls.candidates = cls.exhaustive_learner.learn_candidates(
            cls.test_inputs, cls.exclude_nonterminals
        )

    def test_default_comparison(self):
        """
        Test the default comparison of candidates. It should compare based on recall, precision, and length.
        """
        sorted_candidates = sorted(self.candidates, reverse=True)
        strategy = RecallPriorityLengthFitness()
        sorted_candidates_ = sorted(
            self.candidates, key=lambda c: strategy.evaluate(c), reverse=True
        )
        self.assertEqual(sorted_candidates, sorted_candidates_)

    def test_precision_fitness(self):
        """
        Test the precision fitness strategy. It should evaluate and compare candidates based on precision.
        """
        strategy = PrecisionFitness()
        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(self.candidates, key=lambda c: c.precision(), reverse=True)
        self.assertEqual(sorted_candidates, expected)

    def test_recall_fitness(self):
        """
        Test the recall fitness strategy. It should evaluate and compare candidates based on recall.
        """
        strategy = RecallFitness()
        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(self.candidates, key=lambda c: c.recall(), reverse=True)
        self.assertEqual(sorted_candidates, expected)

    def test_recall_priority_fitness(self):
        """
        Test the recall priority fitness strategy. It should evaluate and compare candidates based on recall and precision.
        """
        strategy = RecallPriorityFitness()
        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(
            self.candidates, key=lambda c: (c.recall(), c.precision()), reverse=True
        )
        self.assertEqual(sorted_candidates, expected)

    def test_f1_score_fitness(self):
        """
        Test the F1 score fitness strategy. It should evaluate and compare candidates based on F1 score.
        """
        strategy = F1ScoreFitness()

        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(
            self.candidates,
            key=lambda c: (
                (2 * c.precision() * c.recall()) / (c.precision() + c.recall()),
                -len(c.formula),
            ),
            reverse=True,
        )
        self.assertEqual(sorted_candidates, expected)


if __name__ == "__main__":
    unittest.main()
