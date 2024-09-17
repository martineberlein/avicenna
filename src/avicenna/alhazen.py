from typing import Set, Optional, List, Iterable

from debugging_framework.fuzzingbook.grammar import Grammar
from debugging_framework.types import OracleType

from .data import Input
from .learning.table import Candidate
from .learning.alhazen_learner import Model, AlhazenLearner
from .generator.alhazen_generator import AlhazenGenerator
from .core import HypothesisInputFeatureDebugger




class Alhazen(HypothesisInputFeatureDebugger):
    """
    Alhazen is a hypothesis-based input feature debugger that uses a decision tree-based learner to explain
    the input features that result in the failure of a program.
    """

    def __init__(
        self,
        grammar: Grammar,
        oracle: OracleType,
        initial_inputs: Iterable[Input | str],
        model: Model = None,
        **kwargs,
    ):
        learner = AlhazenLearner(model=model)
        generator = AlhazenGenerator(grammar=grammar)

        super().__init__(
            grammar,
            oracle,
            initial_inputs,
            learner=learner,
            generator=generator,
            **kwargs,
        )

    def learn_candidates(self, test_inputs: Set[Input]) -> Optional[List[Candidate]]:
        """
        Learn the candidates that result in the failure of a program.
        """
        return self.learner.learn_candidates(test_inputs)

    def generate_test_inputs(self, candidates: List[Candidate]) -> Set[Input]:
        """
        Generate the test inputs based on the learned candidates.
        """
        return self.generator.generate_test_inputs(candidates=candidates)
