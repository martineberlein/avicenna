import unittest
from typing import Set

from isla.language import Formula
from isla.language import ISLaUnparser

from debugging_framework.input.oracle import OracleResult
from avicenna.data import Input
from avicenna.learning.constructor import AtomicFormulaInstantiation
from avicenna.learning.repository import PatternRepository
# from avicenna.learning.heuristic import HeuristicTreePatternCandidateLearner
from avicenna.learning.heuristic_new import HeuristicTreePatternCandidateLearner

from resources.subjects import get_calculator_subject

class TestHeuristicLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.calculator = get_calculator_subject()
        cls.patterns: Set[Formula] = PatternRepository.from_file().get_all()
        cls.test_inputs = set(
            [
                Input.from_str(cls.calculator.get_grammar(), inp, inp_oracle)
                for inp, inp_oracle in [
                    ("sqrt(-901)", OracleResult.FAILING),
                    ("sqrt(-8)", OracleResult.FAILING),
                    ("sqrt(-99)", OracleResult.FAILING),
                    ("sqrt(10)", OracleResult.PASSING),
                    ("sqrt(3)", OracleResult.PASSING),
                    ("cos(1)", OracleResult.PASSING),
                    ("sin(-99)", OracleResult.PASSING),
                    ("tan(-20)", OracleResult.PASSING),
                    ("sqrt(-20)", OracleResult.FAILING),
                ]
            ]
        )
        cls.exclude_nonterminals = {
            "<digits>",
            "<maybe_digits>",
            "<maybe_frac>",
            "<one_nine>",
            "<arith_expr>",
            "<start>",
            "<digit>",
        }
        cls.atomic_candidate_constructor = AtomicFormulaInstantiation(
            cls.calculator.get_grammar(), patterns=list(cls.patterns)
        )

    def test_heuristic_learner(self):
        """
        Test the heuristic learner.
        """
        from isla.fuzzer import GrammarFuzzer
        test_inputs = set()
        for _ in range(500):
            tree = GrammarFuzzer(self.calculator.get_grammar()).fuzz_tree()
            test_inputs.add(Input(tree, self.calculator.oracle(str(tree))[0]))

        #for inp in test_inputs:
            #print(inp.tree, inp.oracle)
        learner = HeuristicTreePatternCandidateLearner(self.calculator.get_grammar())
        learner.learn_candidates(test_inputs, exclude_nonterminals=self.exclude_nonterminals)

        # Get the trained model
        trained_model = learner.model_trainer.get_model()

        # Optionally visualize the decision tree
        learner.model_trainer.visualize_decision_tree(feature_names=learner.data_handler.candidate_names, class_names=["Pass", "Bug"])
        print(learner.data_handler.x_data)
        print(learner.data_handler.y_data)

        print(learner.extract_positive_rules(learner.model_trainer.get_model(), learner.data_handler.candidate_names))


if __name__ == '__main__':
    unittest.main()
