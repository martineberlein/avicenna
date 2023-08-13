import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Set, Callable, Optional, Tuple, Iterable

import islearn.learner
import pandas
from fuzzingbook.Grammars import Grammar, is_valid_grammar
from fuzzingbook.Parser import EarleyParser
from grammar_graph.gg import GrammarGraph
from isla.language import DerivationTree
from isla.language import ISLaUnparser, Formula
from islearn.learner import TruthTable as IslearnTruthTable

from avicenna import feature_extractor
from avicenna.feature_collector import GrammarFeatureCollector
from avicenna.generator import (
    ISLaGrammarBasedGenerator,
    MutationBasedGenerator,
    FuzzingbookBasedGenerator,
)
from avicenna.helpers import (
    Timetable,
)
from avicenna.input import Input
from avicenna.pattern_learner import (
    AvicennaTruthTable,
    AviIslearn,
)
from avicenna.oracle import OracleResult
from avicenna_formalizations import get_pattern_file_path
from avicenna.result_table import TruthTable, TruthTableRow
from avicenna.helpers import time
from avicenna.execution_handler import SingleExecutionHandler, BatchExecutionHandler
from avicenna.report import SingleFailureReport, MultipleFailureReport
from avicenna.logger import LOGGER
from avicenna.monads import Maybe, Exceptional, check_empty

ISLA_GENERATOR_TIMEOUT_SECONDS = 10


class Avicenna(Timetable):
    def __init__(
        self,
        grammar: Grammar,
        oracle: Callable[[Input], OracleResult],
        initial_inputs: List[str],
        working_dir: Path = Path("/tmp").resolve(),
        activated_patterns: List[str] = None,
        max_iterations: int = 10,
        max_excluded_features: int = 3,
        pattern_file: Path = None,
        max_conjunction_size: int = 2,
        use_multi_failure_report: bool = True,
        use_batch_execution: bool = False,
    ):
        super().__init__(working_dir)
        self._start_time = None
        self._activated_patterns = activated_patterns
        self.oracle = oracle
        self._max_iterations: int = max_iterations
        self._top_n: int = max_excluded_features - 1
        self._targeted_start_size: int = 10
        self._iteration = 0
        self._timeout: int = 3600  # timeout in seconds
        self._data = None
        self._all_data = None
        self._learned_invariants: Dict[str, List[float]] = {}
        self._best_candidates: Dict[str, List[float]] = {}

        self._infeasible_constraints: Set = set()
        self._max_conjunction_size = max_conjunction_size

        self.grammar: Grammar = grammar
        assert is_valid_grammar(self.grammar)

        self.collector = GrammarFeatureCollector(self.grammar)
        self.feature_learner = feature_extractor.SHAPRelevanceLearner(
            self.grammar,
            top_n=self._top_n,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
        )

        self._pattern_file = pattern_file if pattern_file else get_pattern_file_path()

        # Islearn
        self._graph = GrammarGraph.from_grammar(grammar)

        self.pattern_learner = AviIslearn(grammar, pattern_file=str(self._pattern_file))

        # TruthTable
        self.truthTable: TruthTable = TruthTable()
        self.precision_truth_table = AvicennaTruthTable()
        self.recall_truth_table = AvicennaTruthTable()

        # All bug triggering inputs
        self.pathological_inputs = set()

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

        test_inputs: Set[Input] = set()
        for inp in initial_inputs:
            try:
                test_inputs.add(
                    Input(
                        DerivationTree.from_parse_tree(
                            next(EarleyParser(self.grammar).parse(inp))
                        )
                    )
                )
            except SyntaxError:
                logging.error(
                    "Avicenna: Could not parse initial inputs with given grammar!"
                )
                sys.exit(-1)

        self.all_inputs: Set[Input] = test_inputs
        self.execution_handler.label(test_inputs, self.report)
        self.best_candidates = set()

    @staticmethod
    def map_to_bool(result: OracleResult) -> bool:
        match result:
            case OracleResult.BUG:
                return True
            case OracleResult.NO_BUG:
                return False
            case _:
                return False

    def generate_more_inputs(self) -> Set[Input]:
        num_failing_inputs = self.get_num_failing_inputs()
        result = self.get_more_inputs(num_failing_inputs)
        return result.value() if result.is_just() else set()

    def get_num_failing_inputs(self) -> int:
        return len(self.report.get_all_failing_inputs())

    def get_more_inputs(self, num_failing_inputs: int) -> Maybe[Set[Input]]:
        generated_inputs: Set[Input] = set()
        if num_failing_inputs < self._targeted_start_size:
            # generator = MutationBasedGenerator(self.grammar, self.oracle, self.all_inputs, True)
            generator = FuzzingbookBasedGenerator(self.grammar)
            for _ in range(20):
                result = generator.generate()
                if result.is_just():
                    generated_inputs.add(result.value())
                else:
                    break
        # If we have generated some inputs, we return them wrapped in a Just monad
        if generated_inputs:
            return Maybe.just(generated_inputs)
        return Maybe.nothing()

    def explain(self) -> List[Tuple[Formula, float, float, float]]:
        new_inputs: Set[Input] = self.all_inputs.union(self.generate_more_inputs())
        while self._do_more_iterations():
            LOGGER.info("Starting Iteration " + str(self._iteration))
            new_inputs = self._loop(new_inputs)

        return self._finalize()

    def _do_more_iterations(self):
        if self._iteration >= self._max_iterations:
            logging.info("Terminate due to maximal iterations reached")
            return False
        self._iteration += 1
        return True

    def add_inputs(self, test_inputs: Set[Input]):
        self.all_inputs.update(test_inputs)
        return test_inputs

    def construct_inputs(self, test_inputs: Set[Input]) -> Set[Input]:
        result: Set[Input] = (
            Exceptional.of(lambda: test_inputs)
            .map(self.assign_label)
            .map(lambda x: {inp for inp in x if inp.oracle != OracleResult.UNDEF})
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
        # first generate and give constraints not inputs
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
            # .map(self.add_negated_constraints)
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


    @staticmethod
    def get_statistics(
        formula: Formula,
        precision_truth_table: IslearnTruthTable,
        recall_truth_table: IslearnTruthTable,
    ) -> Tuple[int, float, float, float, float]:
        recall_results = recall_truth_table[formula].eval_results
        precision_results = precision_truth_table[formula].eval_results
        tp = sum(int(r) for r in recall_results)
        fp = sum(int(r) for r in precision_results)
        return (
            len(precision_results) + len(recall_results),
            tp,
            fp,
            len(recall_results) - tp,
            len(precision_results) - fp,
        )

    def _add_infeasible_constraints(self, constraint):
        self._infeasible_constraints.add(constraint)
        logging.info("Removing infeasible constraint")
        logging.debug(f"Infeasible constraint: {constraint}")

    def _finalize(self) -> List[Tuple[Formula, float, float, float]]:
        logging.info("Avicenna finished")
        logging.info("The best learned failure invariant(s):")

        def meets_criteria(precision_value_, recall_value_):
            return precision_value_ >= 0.9 and recall_value_ >= 0.6

        result = {}
        for idx, precision_row in enumerate(self.precision_truth_table):
            precision_value = 1 - precision_row.eval_result()
            recall_value = self.recall_truth_table[idx].eval_result()

            if meets_criteria(precision_value, recall_value):
                result[precision_row.formula] = (precision_value, recall_value)

        sorted_result = dict(
            sorted(result.items(), key=lambda p: (p[1], -len(p[0])), reverse=True)
        )

        logging.info(
            "\n".join(
                map(
                    lambda x: f"({x[1][0], x[1][1]}) " + ISLaUnparser(x[0]).unparse(),
                    sorted_result.items(),
                )
            )
        )
        return [
            (candidate, stats[0], stats[1])
            for candidate, stats in sorted_result.items()
        ]

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
            generator = ISLaGrammarBasedGenerator(self.grammar)
            for _ in range(1):
                result_ = generator.generate()
                if result_.is_just():
                    generated_inputs.add(result_.value())
                else:
                    break
        return generated_inputs
