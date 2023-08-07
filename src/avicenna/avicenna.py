import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Set, Callable, Optional, Tuple

import pandas
from fuzzingbook.Grammars import Grammar, is_valid_grammar
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from fuzzingbook.Parser import EarleyParser
from grammar_graph.gg import GrammarGraph
from isla.language import DerivationTree
from isla.language import ISLaUnparser, Formula
from isla.solver import ISLaSolver
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
from avicenna.islearn import AvicennaISlearn
from avicenna.oracle import OracleResult
from avicenna_formalizations import get_pattern_file_path
from avicenna.result_table import TruthTable, TruthTableRow
from avicenna.logger import LOGGER
from avicenna.helpers import time
from avicenna.execution_handler import SingleExecutionHandler, BatchExecutionHandler
from avicenna.report import SingleFailureReport, MultipleFailureReport

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
        self._max_excluded_features: int = max_excluded_features - 1
        self._targeted_start_size: int = 10
        self._iteration = 0
        self._timeout: int = 3600  # timeout in seconds
        self._data = None
        self._all_data = None
        self._learned_invariants: Dict[str, List[float]] = {}
        self._best_candidates: Dict[str, List[float]] = {}

        self._feature_table = None
        self._infeasible_constraints: Set = set()
        self._max_conjunction_size = max_conjunction_size

        self.grammar: Grammar = grammar
        assert is_valid_grammar(self.grammar)

        self._inputs: Set[Input] = set()

        self.collector = GrammarFeatureCollector(self.grammar)
        self.feature_learner = feature_extractor.DecisionTreeRelevanceLearner(self.grammar)


        self._pattern_file = (
            pattern_file
            if pattern_file
            else get_pattern_file_path()
        )

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

        # TruthTable
        self.truthTable: TruthTable = TruthTable()

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
            for _ in range(10):
                inp = generator.generate()
                filler_inputs.add(inp)

        return filler_inputs

    @time
    def explain(self) -> List[Tuple[Formula, float, float, float]]:
        LOGGER.info("Starting Avicenna.")
        register_termination(self._timeout)
        try:
            self._start_time = perf_counter()
            new_inputs: Set[Input] = self._setup()
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
        self._inputs.update(test_inputs)

        _ , _ , excluded_features = self.feature_learner.learn(self._inputs)

        excluded_non_terminals: Set[str] = set([feature.non_terminal for feature in excluded_features])
        print(excluded_non_terminals)

        # Run islearn to learn new constraints
        new_candidates: List[Formula] = list(
            self._islearn.learn_failure_invariants(
                self._inputs, excluded_non_terminals
            ).keys()
        )
        # for inv in new_candidates:
        #    print(ISLaUnparser(inv).unparse())

        # Update old candidates
        self.truthTable.evaluate(test_inputs, self._graph)

        # Update new Candidates
        for candidate in new_candidates:
            if hash(candidate) not in self.truthTable.row_hashes:
                self.truthTable.append(
                    TruthTableRow(candidate).evaluate(self._inputs, self._graph)
                )

        statistics = list()
        for row in self.truthTable:
            precision = row.tp / (row.tp + row.fp)
            recall = row.tp / (row.tp + row.fn)
            f1 = (2 * precision * recall) / (precision + recall)
            statistics.append((row.formula, precision, recall, f1))

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

    @time
    def _generate_new_inputs(self, constraints: List[str]):
        logging.info("Generating new inputs to refine constraints.")

        # Exclude all infeasible constraints:
        num_constraints = len(constraints)
        for constraint in constraints:
            if constraint in self._infeasible_constraints:
                constraints.remove(constraint)

        logging.info(
            f"Using {len(constraints)} (of {num_constraints}) constraints to generate new inputs"
        )

        inputs: List[DerivationTree] = []
        for constraint in constraints:
            new_input_trees: List[DerivationTree] = []
            try:
                logging.info(f"Solving: {constraint}")
                solver = ISLaSolver(
                    grammar=self.grammar,
                    formula=constraint,
                    max_number_free_instantiations=10,
                    max_number_smt_instantiations=10,
                    enable_optimized_z3_queries=True,
                    timeout_seconds=ISLA_GENERATOR_TIMEOUT_SECONDS,
                )

                for _ in range(10):
                    gen_inp = solver.solve()
                    new_input_trees.append(gen_inp)
                    logging.info(f"Generated: {gen_inp}")

            except Exception as e:
                logging.info("Error: " + str(e))
            finally:
                if len(new_input_trees) == 0:
                    logging.info(
                        "Removing constraint because no new inputs were generated"
                    )
                    # Solver was not able to generate inputs:
                    self._add_infeasible_constraints(constraint)
                else:
                    inputs = inputs + new_input_trees

        assert len(inputs) != 0, "No new inputs were generated. Exiting."
        logging.info(f"{len(inputs)} new inputs have been generated.")

        new_inputs = set()
        for tree in inputs:
            new_inputs.add(Input(tree=tree))
        return new_inputs

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
                    all_constraints[0:10],
                )
            )
        )

        return all_constraints[0:5]

    def iteration_identifier_map(self):
        return {"iteration": self._iteration}
