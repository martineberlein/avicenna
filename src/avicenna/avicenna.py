import logging
from pathlib import Path
from typing import List, Dict, Set, Callable, Tuple, Iterable

from fuzzingbook.Grammars import Grammar, is_valid_grammar
from isla.language import ISLaUnparser, Formula
from islearn.language import parse_abstract_isla
from islearn.learner import patterns_from_file

from avicenna import feature_extractor
from avicenna.feature_collector import GrammarFeatureCollector
from avicenna.generator import (
    ISLaGrammarBasedGenerator,
    FuzzingbookBasedGenerator,
    MutationBasedGenerator,
)

from avicenna.input import Input
from avicenna.pattern_learner import (
    AvicennaTruthTable,
    AvicennaTruthTableRow,
    AviIslearn,
)
from avicenna_formalizations import get_pattern_file_path
from avicenna.execution_handler import SingleExecutionHandler, BatchExecutionHandler
from avicenna.report import SingleFailureReport, MultipleFailureReport
from avicenna.logger import LOGGER, configure_logging
from avicenna.monads import Exceptional, check_empty
from returns.maybe import Maybe, Some, Nothing

from debugging_framework.oracle import OracleResult


class Avicenna:
    """
    This is the main class of Avicenna. Its top-level methods are:

    :meth: avicenna.Avicenna.explain`
        Use to learn the failure-constraint (debugging constraint).

    :meth: `avicenna.Avicenna.get_equivalent_best_formulas`
        Use to obtain similar (or equivalent) failure constraints.
    """

    def __init__(
        self,
        grammar: Grammar,
        oracle: Callable[[Input], OracleResult],
        initial_inputs: List[str],
        patterns: List[str] = None,
        max_iterations: int = 10,
        top_n_relevant_features: int = 3,
        pattern_file: Path = None,
        max_conjunction_size: int = 2,
        use_multi_failure_report: bool = True,
        use_batch_execution: bool = False,
        log: bool = False,
        feature_learner: feature_extractor.RelevantFeatureLearner = None,
        timeout: int = 3600,
    ):
        """
        The constructor of :class:`~avicenna.Avicenna.` accepts a large number of
        parameters. However, only the :code:`grammar`, the :code:`oracle`, and the :code:`initial_inputs` are required.
        All other parameters are *optional.*

        :param grammar: The underlying grammar as a "Fuzzing Book" dictionary
        :param oracle: The oracle that calls the underlying program and classifies the
            program behavior as `oracle.BUG` or `oracle.NO_BUG`.
        :param initial_inputs: An initial input corpus as strings.
            At least one behavior-triggering input is required.
        :param patterns: A List of patterns that will be used to explain the program behavior.
        :param max_iterations: The maximum number of iteration of the refinement loop.
        :param top_n_relevant_features: The number of *relevant* features during the
            constraint and pattern matching step. The number heavily influences the
            computational overhead of Avicenna. Avicenna will use this number to limit the
            amount of non-terminals that are relevant for describing the program behavior.
        :param pattern_file: A pattern repository with possible patterns.
        :param max_conjunction_size: The maximum number of conjunctions in a learned constraint.
            The higher the number, the higher the computational overhead.
        :param use_multi_failure_report:
        :param use_batch_execution: Toggle batch execution and single file execution.
            During batch execution, the program is called once and given all inputs at the same time.
            During single execution the program gets called for each individual input.
        :param log: Toggle verbose logging.
        :param feature_learner: A constructor class of the RelevantFeatureLearner.
        """

        self._start_time = None
        self.patterns = patterns
        self.oracle = oracle
        self._max_iterations: int = max_iterations
        self._top_n: int = top_n_relevant_features - 1
        self._targeted_start_size: int = 10
        self._iteration = 0
        self._timeout: int = 3600  # timeout in seconds
        self._data = None
        self._all_data = None
        self._learned_invariants: Dict[str, List[float]] = {}
        self._best_candidates: Dict[str, List[float]] = {}
        self.min_precision = 0.6
        self.min_recall = 0.9

        if log:
            configure_logging()
        else:
            # If you want to disable logging when log is set to False
            # Clear root logger handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # Clear avicenna logger handlers
            for handler in LOGGER.handlers[:]:
                LOGGER.removeHandler(handler)

        self._infeasible_constraints: Set = set()
        self._max_conjunction_size = max_conjunction_size

        self.grammar: Grammar = grammar
        assert is_valid_grammar(self.grammar)

        self.collector = GrammarFeatureCollector(self.grammar)
        self.feature_learner = (
            feature_learner
            if feature_learner
            else feature_extractor.SHAPRelevanceLearner(
                self.grammar,
                top_n=self._top_n,
                classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
            )
        )

        self.pattern_file = pattern_file if pattern_file else get_pattern_file_path()
        if not patterns:
            pattern_repo = patterns_from_file(str(self.pattern_file))
            self.patterns = list(pattern_repo.get_all())
        else:
            self.patterns = [
                pattern
                if isinstance(pattern, Formula)
                else parse_abstract_isla(pattern, grammar)
                for pattern in patterns
            ]

        self.pattern_learner = AviIslearn(
            grammar, pattern_file=str(self.pattern_file), patterns=self.patterns
        )

        # TruthTable
        self.precision_truth_table = AvicennaTruthTable()
        self.recall_truth_table = AvicennaTruthTable()

        self.report = (
            MultipleFailureReport()
            if use_multi_failure_report
            else SingleFailureReport()
        )

        self.execution_handler = (
            BatchExecutionHandler(self.oracle)
            if use_batch_execution
            else SingleExecutionHandler(self.oracle)
        )

        self.all_inputs: Set[Input] = (
            Exceptional.of(lambda: initial_inputs)
            .map(self.parse_to_input)
            .map(self.assign_label)
            .reraise()
            .get()
        )

        self.best_candidates = set()

    @staticmethod
    def map_to_bool(result: OracleResult) -> bool:
        match result:
            case OracleResult.FAILING:
                return True
            case OracleResult.PASSING:
                return False
            case _:
                return False

    def generate_more_inputs(self) -> Set[Input]:
        num_failing_inputs = self.get_num_failing_inputs()
        result = self.get_more_inputs(num_failing_inputs)
        return result.value_or(set())

    def get_num_failing_inputs(self) -> int:
        return len(self.report.get_all_failing_inputs())

    def get_more_inputs(self, num_failing_inputs: int) -> Maybe[Set[Input]]:
        generated_inputs: Set[Input] = set()
        if num_failing_inputs < self._targeted_start_size:
            # generator = MutationBasedGenerator(self.grammar, self.oracle, self.all_inputs, True)
            generator = FuzzingbookBasedGenerator(self.grammar)
            for _ in range(50):
                result = generator.generate()
                if result.is_just():
                    generated_inputs.add(result.value())
                else:
                    break
        if generated_inputs:
            return Maybe.from_value(generated_inputs)
        return Nothing

    def explain(self) -> Tuple[Formula, float, float]:
        """
        Attempts to compute a failure diagnosis as an ISLa Formula. It returns that solution, if there is one.
        The result is a tuple with the learned failure constraint and the calculated precision and recall.
        Note that the precision and recall correspond to the values that Avicenna calculated based on a limited subset
        of all possible inputs. Thus, these do not necessarily correspond to the `real` statistical measures.

        :return:
        """
        new_inputs: Set[Input] = self.all_inputs.union(self.generate_more_inputs())
        while self._do_more_iterations():
            new_inputs = self._loop(new_inputs)
        return self.finalize()

    def _do_more_iterations(self):
        if self._iteration >= self._max_iterations:
            LOGGER.info("Terminate due to maximal iterations reached")
            return False
        self._iteration += 1
        LOGGER.info("Starting Iteration " + str(self._iteration))
        return True

    def add_inputs(self, test_inputs: Set[Input]):
        self.all_inputs.update(test_inputs)
        return test_inputs

    def construct_inputs(self, test_inputs: Set[Input]) -> Set[Input]:
        result: Set[Input] = (
            Exceptional.of(lambda: test_inputs)
            .map(self.assign_label)
            .map(lambda x: {inp for inp in x if inp.oracle != OracleResult.UNDEFINED})
            .map(self.assign_feature_vector)
            .map(self.add_inputs)
            .reraise()
            .get()
        )
        return result

    def learn_relevant_features(self) -> List[str]:
        """Learn prominent and correlated features."""
        relevant, correlated, excluded_features = self.feature_learner.learn(
            self.all_inputs
        )
        combined_prominent_non_terminals: Set[str] = set(
            [feature.non_terminal for feature in relevant.union(correlated)]
        )
        return [
            non_terminal
            for non_terminal in self.grammar
            if non_terminal not in combined_prominent_non_terminals
        ]

    def _loop(self, test_inputs: Set[Input]):
        test_inputs = self.construct_inputs(test_inputs)
        exclusion_non_terminals = self.learn_relevant_features()

        new_candidates = self.pattern_learner.learn_failure_invariants(
            test_inputs,
            self.precision_truth_table,
            self.recall_truth_table,
            exclusion_non_terminals,
        )

        new_candidates = new_candidates.keys()

        self.best_candidates = new_candidates
        new_inputs = (
            Exceptional.of(lambda: new_candidates)
            .map(self.add_negated_constraints)
            .map(self.generate_inputs)
            .bind(check_empty)
            .recover(self.generate_inputs_with_grammar_fuzzer)
            .reraise()
            .get()
        )
        LOGGER.info(f"Generated {len(new_inputs)} new inputs.")
        return new_inputs

    def generate_inputs_with_grammar_fuzzer(self, _) -> Set[Input]:
        generator = ISLaGrammarBasedGenerator(self.grammar)
        generated_inputs = set()
        for _ in range(20):
            result_ = generator.generate()
            if result_.is_just():
                generated_inputs.add(result_.value())
            else:
                break
        return generated_inputs

    def _add_infeasible_constraints(self, constraint):
        self._infeasible_constraints.add(constraint)
        logging.info("Removing infeasible constraint")
        logging.debug(f"Infeasible constraint: {constraint}")

    def finalize(self) -> Tuple[Formula, float, float]:
        return (
            self._calculate_best_formula()
            .map(lambda candidates: candidates[0])
            .value_or(None)
        )

    def _calculate_best_formula(self) -> Maybe(List[Tuple[Formula, float, float]]):
        candidates_with_scores = self._gather_candidates_with_scores()
        if not candidates_with_scores:
            return Nothing
        best_candidates = self._get_best_candidates(candidates_with_scores)
        return Some(best_candidates) if best_candidates else Nothing

    def get_equivalent_best_formulas(self) -> List[Tuple[Formula, float, float]]:
        return (
            self._calculate_best_formula()
            .bind(
                lambda candidates: Some(candidates[1:])
                if len(candidates) > 1
                else Nothing
            )
            .value_or(None)
        )

    def _gather_candidates_with_scores(self) -> List[Tuple[Formula, float, float]]:
        def meets_criteria(precision_value_, recall_value_):
            return (
                precision_value_ >= self.min_precision
                and recall_value_ >= self.min_recall
            )

        candidates_with_scores = []

        for idx, precision_row in enumerate(self.precision_truth_table):
            assert isinstance(precision_row, AvicennaTruthTableRow)
            precision_value = 1 - precision_row.eval_result()
            recall_value = self.recall_truth_table[idx].eval_result()

            if meets_criteria(precision_value, recall_value):
                candidates_with_scores.append(
                    (precision_row.formula, precision_value, recall_value)
                )

        candidates_with_scores.sort(
            key=lambda x: (x[1], x[2], -len(x[0])), reverse=True
        )

        return candidates_with_scores

    @staticmethod
    def _get_best_candidates(
        candidates_with_scores: List[Tuple[Formula, float, float]]
    ) -> List[Tuple[Formula, float, float]]:
        top_precision, top_recall = (
            candidates_with_scores[0][1],
            candidates_with_scores[0][2],
        )

        return [
            candidate
            for candidate in candidates_with_scores
            if candidate[1] == top_precision and candidate[2] == top_recall
        ]

    @staticmethod
    def _log_best_candidates(best_candidates: List[Tuple[Formula, float, float]]):
        LOGGER.info(
            "\n".join(
                [
                    f"({candidate[1], candidate[2]}) "
                    + ISLaUnparser(candidate[0]).unparse()
                    for candidate in best_candidates
                ]
            )
        )

    def parse_to_input(self, test_inputs: Iterable[str]) -> Set[Input]:
        return set([Input.from_str(self.grammar, inp_) for inp_ in test_inputs])

    def assign_label_single(self, test_inputs: Set[Input]) -> Set[Input]:
        for inp_ in test_inputs:
            inp_.oracle = self.oracle(inp_)
        return test_inputs

    def assign_label(self, test_inputs: Set[Input]) -> Set[Input]:
        self.execution_handler.label(test_inputs)
        return test_inputs

    def assign_feature_vector(self, test_inputs: Set[Input]) -> Set[Input]:
        collector = GrammarFeatureCollector(self.grammar)
        for inp_ in test_inputs:
            inp_.features = collector.collect_features(inp_)
        return test_inputs

    def generate_inputs(self, candidate_set):
        generated_inputs = set()
        for _ in candidate_set:
            generator = MutationBasedGenerator(
                self.grammar, self.oracle, self.all_inputs, yield_negative=True
            )
            for _ in range(1):
                result_ = generator.generate()
                if result_.is_just():
                    generated_inputs.add(result_.value())
                else:
                    break
        return generated_inputs

    @staticmethod
    def add_negated_constraints(candidates: Set[Formula]) -> Set[Formula]:
        negated_constraints = [-c for c in candidates]
        return candidates.union(negated_constraints)
