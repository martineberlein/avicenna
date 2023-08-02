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
from sklearn.metrics import precision_score, recall_score, f1_score

from avicenna.feature_collector import Collector
from avicenna.features import FeatureWrapper, STANDARD_FEATURES
from avicenna.generator import generate_inputs
from avicenna.helpers import (
    Timetable,
    register_termination,
    CustomTimeout,
    time,
    instantiate_learner,
    constraint_eval,
)
from avicenna.input import Input
from avicenna.islearn import AvicennaISlearn
from avicenna.learner import InputElementLearner
from avicenna.oracle import OracleResult
from avicenna_formalizations import get_pattern_file_path
from avicenna.result_table import TruthTable, TruthTableRow

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
        max_relevant_features: int = 3,
        pattern_file: Path = None,
        max_conjunction_size: int = 2,
    ):
        super().__init__(working_dir)
        self._start_time = None
        self._activated_patterns = activated_patterns
        self._oracle = oracle
        self._max_iterations: int = max_iterations
        self._max_relevant_features: int = max_relevant_features 
        self._targeted_start_size: int = 10
        self._iteration = 0
        self._timeout: int = 3600  # timeout in seconds
        self._data = None
        self._all_data = None
        self._learned_invariants: Dict[str, List[float]] = {}
        self._best_candidates: Dict[str, List[float]] = {}
        if pattern_file is not None:
            logging.info(f"Loading pattern file from location: {str(pattern_file)}")
            self._pattern_file = pattern_file
        else:
            logging.info(f"Loading default pattern file: {get_pattern_file_path()}")
            self._pattern_file = get_pattern_file_path()
        self._feature_table = None
        self._infeasible_constraints: Set = set()
        self._max_conjunction_size = max_conjunction_size

        self._grammar: Grammar = grammar
        assert is_valid_grammar(self._grammar)

        self._initial_inputs: List[str] = initial_inputs
        self._inputs: Set[Input] = set()
        self._new_inputs: Set[Input] = set()

        # Syntactic Feature Collection
        self._syntactic_features: Set[FeatureWrapper] = STANDARD_FEATURES
        self._collector: Collector = Collector(self._grammar, self._syntactic_features)
        self._all_features = self._collector.get_all_features()
        self._feature_names = [f.name for f in self._all_features]

        # Input Element Learner
        self._input_element_learner = InputElementLearner(
            self._grammar, self._oracle, self._max_relevant_features
        )

        # Islearn
        self._graph = GrammarGraph.from_grammar(grammar)

        def dummy_oracle(inp_):
            return True if self._oracle(inp_) == OracleResult.BUG else False

        self._dummy_oracle = dummy_oracle
        self._islearn: AvicennaISlearn = instantiate_learner(
            grammar=self._grammar,
            oracle=dummy_oracle,
            activated_patterns=self._activated_patterns,
            pattern_file=self._pattern_file,
            max_conjunction_size=self._max_conjunction_size # added max_conj_size
        )

        # TruthTable
        self.truthTable: TruthTable = TruthTable()

        # All bug triggering inputs
        self.pathological_inputs = set()

    def _setup(self) -> Set[Input]:
        """
        This function parses the given initial input files and obtains the execution label ("BUG"/"NO_BUG")
        :return:
        """
        test_inputs: Set[Input] = set()
        for inp in self._initial_inputs:
            try:
                test_inputs.add(
                    Input(
                        DerivationTree.from_parse_tree(
                            next(EarleyParser(self._grammar).parse(inp))
                        )
                    )
                )
            except SyntaxError:
                logging.error(
                    "Avicenna: Could not parse initial inputs with given grammar!"
                )
                sys.exit(-1)

        for inp in test_inputs:
            inp.oracle = self._oracle(inp)
            print(inp, inp.oracle)

        num_bug_inputs = len(
            [inp for inp in test_inputs if inp.oracle == OracleResult.BUG]
        )

        if num_bug_inputs < self._targeted_start_size:
            fuzzer = GrammarFuzzer(grammar=self._grammar)
            for _ in range(10):
                test_inputs.add(
                    Input(DerivationTree.from_parse_tree(fuzzer.fuzz_tree()))
                )
        return test_inputs

    def execute(self) -> List[Tuple[Formula, float, float, float]]:
        logging.info("Starting AVICENNA.")
        register_termination(self._timeout)
        try:
            self._start_time = perf_counter()
            new_inputs: Set[Input] = self._setup()
            while self._do_more_iterations():
                logging.info("Starting Iteration " + str(self._iteration))
                new_inputs = self._loop(new_inputs)
                self._iteration = self._iteration + 1
        except CustomTimeout:
            logging.exception("Terminate due to timeout")
        return self._finalize()

    def _do_more_iterations(self):
        if self._iteration >= self._max_iterations:
            logging.info("Terminate due to maximal iterations reached")
            return False
        return True

    def _loop(self, test_inputs: Set[Input]):
        # obtain labels, execute samples (Initial Step, Activity 5)
        for inp in test_inputs:
            label = self._oracle(inp)
            if label == OracleResult.BUG:
                self.pathological_inputs.add(inp)
            inp.oracle = label

        # collect features from the new samples (Activity 1)
        for inp in test_inputs:
            inp.features = self._collector.collect_features(inp)

        # add to global list
        self._inputs.update(test_inputs)

        # Extract the most important non-terminals that are responsible for the program behavior
        excluded_non_terminals: Set[str] = self._get_exclusion_set(self._inputs)

        # Run islearn to learn new constraints
        new_candidates: List[Formula] = list(
            self._islearn.learn_failure_invariants(
                self._inputs, excluded_non_terminals
            ).keys()
        )
        #for inv in new_candidates:
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
            f1 = (2*precision*recall) / (precision + recall)
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
        fuzzer = GrammarFuzzer(grammar=self._grammar)
        for _ in range(20):
            test_inputs.add(Input(DerivationTree.from_parse_tree(fuzzer.fuzz_tree())))
        return test_inputs

    def _get_exclusion_set(self, test_inputs: Set[Input]) -> Set[str]:
        logging.info("Determining the most important non-terminals.")

        learner = InputElementLearner(
            grammar=self._grammar,
            oracle=self._oracle,
            max_relevant_features=self._max_relevant_features,
        )
        learner.learn(test_inputs)
        relevant, irrelevant = learner.get_exclusion_sets()

        logging.info(f"Determined: {relevant} to be the most relevant features.")
        logging.info(f"Excluding: {irrelevant} from candidate consideration.")

        return irrelevant

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
                    grammar=self._grammar,
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

    def _get_best_constraints(self):
        all_constraints = []
        for row in self.truthTable:
            precision = row.tp / (row.tp + row.fp)
            recall = row.tp / (row.tp + row.fn)
            f1 = (2*precision*recall) / (precision + recall)
            all_constraints.append((row.formula, precision, recall, f1 ))

    def _finalize(self) -> List[Tuple[Formula, float, float, float]]:
        logging.info("Avicenna finished")
        logging.info("The best learned failure invariant(s):")

        all_constraints = []
        for row in self.truthTable:
            precision = row.tp / (row.tp + row.fp)
            recall = row.tp / (row.tp + row.fn)
            f1 = (2*precision*recall) / (precision + recall)
            all_constraints.append((row.formula, precision, recall, f1 ))

        all_constraints.sort(key=lambda x: x[3], reverse=True)

        logging.info("\n".join(map(
            lambda p: f"({p[1], p[2], p[3]}): " + ISLaUnparser(p[0]).unparse(),all_constraints[0:10])))

        return all_constraints[0:5]

    def iteration_identifier_map(self):
        return {"iteration": self._iteration}
