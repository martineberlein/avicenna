import unittest

from avicenna.learning.candidate import Candidate
from avicenna.learning.metric import PrecisionFitness, RecallFitness, RecallPriorityFitness


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
            Candidate(formula="formula3", precision=0.95, recall=0.7),
        ]

    def test_default_comparison(self):
        """
        Test the default comparison of candidates.
        Candidates are compared based on RecallPriorityFitness.
        """
        cand = Candidate(formula="f1", precision=0.60, recall=0.85)
        self.assertLess(cand, Candidate(formula="formula2", precision=0.70, recall=0.95))
        self.assertLess(cand, Candidate(formula="formula3", precision=0.80, recall=0.85))
        self.assertLess(cand, Candidate(formula="f", precision=0.60, recall=0.85))

        self.assertGreater(cand, Candidate(formula="f", precision=0.60, recall=0.80))
        self.assertGreater(cand, Candidate(formula="f", precision=0.99, recall=0.84))
        self.assertGreater(cand, Candidate(formula="formula100", precision=0.60, recall=0.85))


if __name__ == '__main__':
    unittest.main()
