import unittest
from typing import Set

from isla.language import Formula
from isla.language import ISLaUnparser

from debugging_framework.input.oracle import OracleResult
from avicenna.data import Input
from avicenna.learning.constructor import AtomicCandidateInstantiation
from avicenna.learning.repository import PatternRepository
from avicenna.learning.table import Candidate

from resources.subjects import get_calculator_subject


class TestCandidates(unittest.TestCase):
    """Test the Candidate"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.calculator = get_calculator_subject()
        cls.patterns: Set[Formula] = PatternRepository.from_file().get_all()

        cls.positive_inputs = set(
            [
                Input.from_str(cls.calculator.get_grammar(), inp, inp_oracle)
                for inp, inp_oracle in [
                    ("sqrt(-2)", OracleResult.FAILING),
                    ("sqrt(-478)", OracleResult.FAILING),
                    ("sqrt(-65)", OracleResult.FAILING),
                    ("sqrt(-98.971)", OracleResult.FAILING),
                    ("sqrt(-8.289)", OracleResult.FAILING),
                    ("sqrt(-2.78)", OracleResult.FAILING),
                ]
            ]
        )
        cls.exclude_nonterminals = {
            "<digits>",
            "<maybe_digits>",
            "<onenine>",
            "<arith_expr>",
            "<start>",
            "<digit>",
        }
        cls.atomic_candidate_constructor = AtomicCandidateInstantiation(
            cls.calculator.get_grammar(), patterns=list(cls.patterns)
        )

    @staticmethod
    def debug_candidates(candidates: Set[Candidate]):
        print("Number of candidates: ", len(candidates))
        for candidate in candidates:
            print(ISLaUnparser(candidate.formula).unparse(), end="\n\n")

    def test_atomic_candidates(self):
        """
        Test the atomic candidate instantiation.
        """
        candidates = self.atomic_candidate_constructor.construct_candidates(
            self.positive_inputs
        )
        self.assertEqual(len(candidates), 385)
        self.debug_candidates(candidates)
        self.assertTrue(candidates)

    def test_atomic_candidates_with_exclude_non_terminals(self):
        """
        Test the atomic candidate instantiation with excluded non-terminals.
        """
        candidates = self.atomic_candidate_constructor.construct_candidates(
            self.positive_inputs, self.exclude_nonterminals
        )
        self.assertEqual(len(candidates), 136)
        self.debug_candidates(candidates)
        self.assertTrue(candidates)


if __name__ == "__main__":
    unittest.main()
