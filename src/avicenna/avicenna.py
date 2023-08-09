import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Set, Callable, Optional, Tuple

import islearn.learner
import pandas
from fuzzingbook.Grammars import Grammar, is_valid_grammar
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from fuzzingbook.Parser import EarleyParser
from grammar_graph.gg import GrammarGraph
from isla.language import DerivationTree
from isla.language import ISLaUnparser, Formula
from isla.solver import ISLaSolver
from islearn.learner import TruthTable as IslearnTruthTable

# from sklearn.metrics import precision_score, recall_score, f1_score

from avicenna import feature_extractor
from avicenna.feature_collector import GrammarFeatureCollector
from avicenna.generator import SimpleGenerator
from avicenna.helpers import (
    Timetable,
    register_termination,
    CustomTimeout,
    instantiate_learner,
)
from avicenna.input import Input
from avicenna.islearn import AvicennaISlearn, AvicennaTruthTable, AvicennaTruthTableRow, AviIslearn
from avicenna.oracle import OracleResult
from avicenna_formalizations import get_pattern_file_path
from avicenna.result_table import TruthTable, TruthTableRow
from avicenna.helpers import time
from avicenna.execution_handler import SingleExecutionHandler, BatchExecutionHandler
from avicenna.report import SingleFailureReport, MultipleFailureReport
from avicenna.logger import LOGGER

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
        self._oracle = oracle
        self._max_iterations: int = max_iterations
        self._top_n: int = max_excluded_features -1
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
            self.grammar, top_n=self._top_n, classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner
        )

        self._pattern_file = pattern_file if pattern_file else get_pattern_file_path()

        # Islearn
        self._graph = GrammarGraph.from_grammar(grammar)

        def boolean_oracle(inp_):
            return self.map_to_bool(self._oracle(inp_))

        self._boolean_oracle = boolean_oracle
        self._islearn: AvicennaISlearn = instantiate_learner(
            grammar=self.grammar,
            oracle=boolean_oracle,
            activated_patterns=self._activated_patterns,
            pattern_file=self._pattern_file,
        )

        self.pattern_learner = AviIslearn(
            grammar,
            pattern_file=str(self._pattern_file)
        )

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
            BatchExecutionHandler(self._oracle)
            if use_batch_execution
            else SingleExecutionHandler(self._oracle)
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

    @staticmethod
    def map_to_bool(result: OracleResult) -> bool:
        match result:
            case OracleResult.BUG:
                return True
            case OracleResult.NO_BUG:
                return False
            case _:
                return False

    def _setup(self) -> Set[Input]:
        """
        This function parses the given initial input files and obtains the execution label ("BUG"/"NO_BUG")
        :return:
        """
        num_failing_inputs = len(self.report.get_all_failing_inputs())

        filler_inputs: Set[Input] = set()
        generator = SimpleGenerator(self.grammar)
        if num_failing_inputs < self._targeted_start_size:
            for _ in range(100):
                inp = generator.generate()
                filler_inputs.add(inp)

        return filler_inputs

    @time
    def explain(self) -> List[Tuple[Formula, float, float, float]]:
        LOGGER.info("Starting Avicenna.")
        register_termination(self._timeout)
        try:
            self._start_time = perf_counter()
            new_inputs: Set[Input] = self.all_inputs.union(self._setup())
            while self._do_more_iterations():
                LOGGER.info("Starting Iteration " + str(self._iteration))
                new_inputs = self._loop(new_inputs)
                self._iteration = self._iteration + 1
        except CustomTimeout:
            LOGGER.exception("Terminate due to timeout")
        return self._finalize()

    def _do_more_iterations(self):
        if self._iteration >= self._max_iterations:
            logging.info("Terminate due to maximal iterations reached")
            return False
        return True

    @time
    def _loop(self, test_inputs: Set[Input]):
        # obtain labels, execute samples (Initial Step, Activity 5)
        self.execution_handler.label(test_inputs, self.report)

        # collect features from the new samples (Activity 1)
        for inp in test_inputs:
            inp.features = self.collector.collect_features(inp)

        # add to global list
        self.all_inputs.update(test_inputs)

        prom, corr, excluded_features = self.feature_learner.learn(self.all_inputs)

        combined_prominent_non_terminals: Set[str] = set(
            [feature.non_terminal for feature in prom.union(corr)]
        )
        exclusion_non_terminals = [
            non_terminal
            for non_terminal in self.grammar
            if non_terminal not in combined_prominent_non_terminals
        ]
        print(combined_prominent_non_terminals)
        print(exclusion_non_terminals)

        # new_candidates, precision_truth_table, recall_truth_table = self._islearn.learn_failure_invariants(
        #     self.all_inputs, exclusion_non_terminals
        #)
        new_candidates, self.precision_truth_table, self.recall_truth_table = self.pattern_learner.learn_failure_invariants(
            test_inputs, self.precision_truth_table, self.recall_truth_table, exclusion_non_terminals
        )

        new_candidates = new_candidates.keys()

        LOGGER.info("Eval new stuff")
        # Update new Candidates
        for candidate in new_candidates:
            if hash(candidate) not in self.truthTable.row_hashes:
                self.truthTable.append(TruthTableRow(candidate).set_results(*self.get_statistics(candidate, self.precision_truth_table, self.recall_truth_table)))
            else:
                self.truthTable[candidate].evaluate(test_inputs, self._graph)

        untouched_formulas = [row.formula for row in self.truthTable if row.formula not in set(new_candidates)]
        for formula in untouched_formulas:
            self.truthTable[formula].evaluate(test_inputs, self._graph)

        for row in self.truthTable:
            _, tp, fp, fn, _ = row.eval_result()
            if (tp/(tp + fp) < 0.6) or tp/(tp + fn) < 0.9:
                self.truthTable.remove(row)

        # negate Constraints
        # TODO
        # 1. negate Constraints
        # 1.2 Check for infeasible constraints
        # 2. safe grammar graph
        # 3. for candidate in new_candidates:
        #   Only use top 5 constraints

        # Generate new Inputs
        test_inputs = set()
        fuzzer = GrammarFuzzer(grammar=self.grammar)
        for _ in range(20):
            test_inputs.add(Input(DerivationTree.from_parse_tree(fuzzer.fuzz_tree())))
        return test_inputs

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

    @time
    def _get_best_constraints(self):
        all_constraints = []
        for row in self.truthTable:
            precision = row.tp / (row.tp + row.fp)
            recall = row.tp / (row.tp + row.fn)
            f1 = (2 * precision * recall) / (precision + recall)
            all_constraints.append((row.formula, precision, recall, f1))

    @time
    def _finalize(self) -> List[Tuple[Formula, float, float, float]]:
        logging.info("Avicenna finished")
        logging.info("The best learned failure invariant(s):")

        all_constraints = []
        for row in self.truthTable:
            precision = row.tp / (row.tp + row.fp)
            recall = row.tp / (row.tp + row.fn)
            f1 = (2 * precision * recall) / (precision + recall)
            all_constraints.append((row.formula, precision, recall, f1))

        all_constraints.sort(key=lambda x: x[3], reverse=True)

        logging.info(
            "\n".join(
                map(
                    lambda p: f"({p[1], p[2], p[3]}): " + ISLaUnparser(p[0]).unparse(),
                    all_constraints[0:30],
                )
            )
        )

        return all_constraints[0:5]

    def iteration_identifier_map(self):
        return {"iteration": self._iteration}
