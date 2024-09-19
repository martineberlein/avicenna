import logging
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Set, List, Union
import time

from debugging_framework.fuzzingbook.grammar import Grammar, is_valid_grammar
from debugging_framework.types import OracleType

from .data import Input
from .learning.learner import CandidateLearner
from .learning.table import Candidate
from .learning.exhaustive import ExhaustivePatternCandidateLearner
from .learning.metric import FitnessStrategy, RecallPriorityStringLengthFitness
from .generator.generator import Generator, ISLaGrammarBasedGenerator
from .runner.execution_handler import ExecutionHandler, SingleExecutionHandler
from .logger import configure_logging


class InputFeatureDebugger(ABC):
    """
    Interface for debugging input features that result in the failure of a program.
    """

    def __init__(
        self,
        grammar: Grammar,
        oracle: OracleType,
        initial_inputs: Union[Iterable[str], Iterable[Input]],
        enable_logging: bool = False,
    ):
        """
        Initialize the input feature debugger with a grammar, oracle, and initial inputs.
        """
        assert is_valid_grammar(grammar)
        self.initial_inputs = initial_inputs
        self.grammar = grammar
        self.oracle = oracle
        configure_logging(enable_logging)

    @abstractmethod
    def explain(self, *args, **kwargs):
        """
        Explain the input features that result in the failure of a program.
        """
        raise NotImplementedError()


class HypothesisInputFeatureDebugger(InputFeatureDebugger, ABC):
    """
    A hypothesis-based input feature debugger.
    """

    def __init__(
        self,
        grammar: Grammar,
        oracle: OracleType,
        initial_inputs: Union[Iterable[str], Iterable[Input]],
        learner: Optional[CandidateLearner] = None,
        generator: Optional[Generator] = None,
        runner: Optional[ExecutionHandler] = None,
        timeout_seconds: Optional[int] = None,
        max_iterations: Optional[int] = 10,
        **kwargs,
    ):
        """
        Initialize the hypothesis-based input feature debugger with a grammar, oracle, initial inputs,
        learner, generator, and runner.
        """
        super().__init__(grammar, oracle, initial_inputs, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.strategy = RecallPriorityStringLengthFitness()
        self.learner: CandidateLearner = (
            learner if learner else ExhaustivePatternCandidateLearner(self.grammar)
        )
        self.generator: Generator = (
            generator if generator else ISLaGrammarBasedGenerator(self.grammar)
        )
        self.runner: ExecutionHandler = (
            runner if runner else SingleExecutionHandler(self.oracle)
        )

    def set_runner(self, runner: ExecutionHandler):
        """
        Set the runner for the hypothesis-based input feature debugger.
        """
        self.runner = runner

    def set_learner(self, learner: Optional[CandidateLearner]):
        """
        Set the learner for the hypothesis-based input feature debugger.
        """
        self.learner = (
            learner if learner else ExhaustivePatternCandidateLearner(self.grammar)
        )

    def set_generator(self, generator: Generator):
        """
        Set the generator for the hypothesis-based input feature debugger.
        """
        self.generator = generator

    def set_timeout(self) -> Optional[float]:
        """
        Set the timeout for the hypothesis-based input feature debugger.
        Returns the start time if the timeout is set, otherwise None.
        """
        if self.timeout_seconds is not None:
            return int(time.time())
        return None

    def check_timeout_reached(self, start_time) -> bool:
        """
        Check if the timeout has been reached.
        """
        if self.timeout_seconds is None:
            return False
        return time.time() - start_time >= self.timeout_seconds

    def check_iterations_reached(self, iteration) -> bool:
        """
        Check if the maximum number of iterations has been reached.
        """
        return iteration >= self.max_iterations

    def check_iteration_limits(self, iteration, start_time) -> bool:
        if self.check_iterations_reached(iteration):
            return False
        if self.check_timeout_reached(start_time):
            return False
        return True

    def explain(self) -> Optional[List[Candidate]]:
        """
        Explain the input features that result in the failure of a program.
        """
        iteration = 0
        start_time = self.set_timeout()
        try:
            test_inputs: Set[Input] = self.prepare_test_inputs()

            while self.check_iteration_limits(iteration, start_time):
                new_test_inputs = self.hypothesis_loop(test_inputs)
                test_inputs.update(new_test_inputs)

                iteration += 1
        except TimeoutError as e:
            logging.error(e)
        finally:
            return self.get_best_candidates()

    def prepare_test_inputs(self) -> Set[Input]:
        """
        Prepare the input feature debugger.
        """
        test_inputs: Set[Input] = self.get_test_inputs_from_strings(self.initial_inputs)
        test_inputs = self.run_test_inputs(test_inputs)
        self.check_initial_conditions(test_inputs)
        return test_inputs

    def hypothesis_loop(self, test_inputs: Set[Input]) -> Set[Input]:
        """
        The main loop of the hypothesis-based input feature debugger.
        """
        candidates = self.learn_candidates(test_inputs)
        negated_candidates = self.negate_candidates(candidates)
        inputs = self.generate_test_inputs(candidates+negated_candidates)
        labeled_test_inputs = self.run_test_inputs(inputs)
        return labeled_test_inputs

    def learn_candidates(self, test_inputs: Set[Input]) -> Optional[List[Candidate]]:
        """
        Learn the candidates (failure diagnoses) from the test inputs.
        """
        return self.learner.learn_candidates(test_inputs)

    @staticmethod
    def negate_candidates(candidates: List[Candidate]):
        """
        Negate the learned candidates.
        """
        negated_candidates = []
        for candidate in candidates:
            negated_candidates.append(-candidate)
        return negated_candidates

    def generate_test_inputs(self, candidates: List[Candidate]) -> Set[Input]:
        """
        Generate the test inputs based on the learned candidates.
        """
        return self.generator.generate_test_inputs(candidates=candidates)

    def run_test_inputs(self, test_inputs: Set[Input]) -> Set[Input]:
        """
        Run the test inputs.
        """
        return self.runner.label(test_inputs=test_inputs)

    def get_best_candidates(
        self, strategy: Optional[FitnessStrategy] = None
    ) -> Optional[List[Candidate]]:
        """
        Return the best candidate.
        """
        strategy = strategy if strategy else self.strategy
        candidates = self.learner.get_best_candidates()
        sorted_candidates = sorted(candidates, key=lambda c: strategy.evaluate(c), reverse=True) if candidates else []
        return sorted_candidates

    def get_test_inputs_from_strings(self, inputs: Iterable[str]) -> Set[Input]:
        """
        Convert a list of input strings to a set of Input objects.
        """
        return set([Input.from_str(self.grammar, inp, None) for inp in inputs])

    @staticmethod
    def check_initial_conditions(test_inputs: Set[Input]):
        """
        Check the initial conditions for the input feature debugger.
        Raises a ValueError if the conditions are not met.
        """

        has_failing = any(inp.oracle.is_failing() for inp in test_inputs)
        has_passing = any(not inp.oracle.is_failing() for inp in test_inputs)

        if not (has_failing and has_passing):
            raise ValueError("The initial inputs must contain at least one failing and one passing input.")
