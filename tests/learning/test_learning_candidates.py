import unittest
from avicenna.learning.candidate import Candidate
from avicenna.learning.metric import (
    PrecisionFitness,
    RecallFitness,
    RecallPriorityFitness,
    F1ScoreFitness,
    RecallPriorityLengthFitness,
)


class TestCandidates(unittest.TestCase):
    """Test the Candidate"""

    @classmethod
    def setUpClass(cls):
        cls.candidates = [
            Candidate(formula="formula1", precision=0.75, recall=0.85),
            Candidate(formula="formula2", precision=0.70, recall=0.95),
            Candidate(formula="formula3", precision=0.80, recall=0.85),
            Candidate(formula="f1", precision=0.60, recall=0.85),
            Candidate(formula="formula3", precision=0.62, recall=0.85),
            Candidate(formula="f", precision=0.60, recall=0.85),
            Candidate(formula="formula100", precision=0.60, recall=0.85),
            Candidate(formula="formula3", precision=0.95, recall=0.70),
        ]
        cls.cand = Candidate(formula="f1", precision=0.60, recall=0.85)

    def test_default_comparison(self):
        """
        Test the default comparison of candidates. It should compare based on recall, precision, and length.
        """
        self.assertLess(
            self.cand, Candidate(formula="formula2", precision=0.70, recall=0.95)
        )
        self.assertLess(
            self.cand, Candidate(formula="formula3", precision=0.80, recall=0.85)
        )
        self.assertLess(self.cand, Candidate(formula="f", precision=0.60, recall=0.85))

        self.assertGreater(
            self.cand, Candidate(formula="f", precision=0.60, recall=0.80)
        )
        self.assertGreater(
            self.cand, Candidate(formula="f", precision=0.99, recall=0.84)
        )
        self.assertGreater(
            self.cand, Candidate(formula="formula100", precision=0.60, recall=0.85)
        )

        sorted_candidates = sorted(self.candidates, reverse=True)
        strategy = RecallPriorityLengthFitness()
        sorted_candidates_ = sorted(
            self.candidates, key=strategy.evaluate, reverse=True
        )
        self.assertEqual(sorted_candidates, sorted_candidates_)

    def test_precision_fitness(self):
        """
        Test the precision fitness strategy. It should evaluate and compare candidates based on precision.
        """
        strategy = PrecisionFitness()
        self.assertEqual(strategy.evaluate(self.cand), 0.60)
        self.assertEqual(
            strategy.compare(
                self.cand, Candidate(formula="f", precision=0.60, recall=0.85)
            ),
            0,
        )

        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(self.candidates, key=lambda c: c.precision, reverse=True)
        self.assertEqual(sorted_candidates, expected)

    def test_recall_fitness(self):
        """
        Test the recall fitness strategy. It should evaluate and compare candidates based on recall.
        """
        strategy = RecallFitness()
        self.assertEqual(strategy.evaluate(self.cand), 0.85)
        self.assertEqual(
            strategy.compare(
                self.cand, Candidate(formula="f", precision=0.60, recall=0.85)
            ),
            0,
        )

        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(self.candidates, key=lambda c: c.recall, reverse=True)
        self.assertEqual(sorted_candidates, expected)

    def test_recall_priority_fitness(self):
        """
        Test the recall priority fitness strategy. It should evaluate and compare candidates based on recall and precision.
        """
        strategy = RecallPriorityFitness()
        self.assertEqual(strategy.evaluate(self.cand), (0.85, 0.60))
        self.assertEqual(
            strategy.compare(
                self.cand, Candidate(formula="f", precision=0.60, recall=0.85)
            ),
            0,
        )

        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(
            self.candidates, key=lambda c: (c.recall, c.precision), reverse=True
        )
        self.assertEqual(sorted_candidates, expected)

    def test_f1_score_fitness(self):
        """
        Test the F1 score fitness strategy. It should evaluate and compare candidates based on F1 score.
        """
        cand_f1 = Candidate(formula="formula1", precision=0.6, recall=0.7)
        strategy = F1ScoreFitness()
        self.assertEqual(strategy.evaluate(cand_f1), (0.6461538461538462, -8))
        self.assertNotEqual(
            strategy.compare(
                cand_f1, Candidate(formula="f", precision=0.6, recall=0.7)
            ),
            0,
        )
        self.assertEqual(
            strategy.compare(
                cand_f1, Candidate(formula="formula1", precision=0.6, recall=0.7)
            ),
            0,
        )

        sorted_candidates = sorted(self.candidates, key=strategy.evaluate, reverse=True)
        expected = sorted(
            self.candidates,
            key=lambda c: (
                (2 * c.precision * c.recall) / (c.precision + c.recall),
                -len(c.formula),
            ),
            reverse=True,
        )
        self.assertEqual(sorted_candidates, expected)


if __name__ == "__main__":
    unittest.main()
