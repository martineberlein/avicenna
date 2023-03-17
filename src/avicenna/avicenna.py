from typing import List, Dict, Iterable, Union, Set, Callable, Tuple
import logging
import sys
from time import perf_counter
import pandas
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import chain
from pathlib import Path

from isla import language
from isla.language import ISLaUnparser
from isla.solver import ISLaSolver
from isla.language import DerivationTree
from fuzzingbook.Parser import EarleyParser
from fuzzingbook.Grammars import Grammar, is_valid_grammar

from avicenna.helpers import (
    Timetable,
    register_termination,
    CustomTimeout,
    time,
    run_islearn,
    constraint_eval,
)
from avicenna.generator import Generator, generate_inputs
from avicenna.learner import InputElementLearner
from avicenna_formalizations import get_pattern_file_path
from avicenna.input import Input
from avicenna.oracle import OracleResult

ISLA_GENERATOR_TIMEOUT_SECONDS = 10


class Avicenna(Timetable):
    def __init__(
            self,
            grammar: Grammar,
            prop: Callable[[DerivationTree], bool],
            initial_inputs: List[str],
            working_dir: Path = Path("/tmp").resolve(),
            activated_patterns: List[str] = None,
            max_iterations: int = 10,
            max_excluded_features: int = 3,
            pattern_file: Path = None,
            max_conjunction_size: int = 2,
    ):

        super().__init__(working_dir)
        self._start_time = None
        self._activated_patterns = activated_patterns
        self._prop = prop
        self._max_iterations: int = max_iterations
        self._max_excluded_features: int = max_excluded_features - 1
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

    def _execute_input(self, test_input: Input) -> OracleResult:
        """
        A wrapper function that wraps the user defined prop (evaluation function). This wrapper returns an OracleResult.
        :param test_input:
        :return:
        """
        return OracleResult.BUG if self._prop(test_input.tree) else OracleResult.NO_BUG

    def _setup(self):
        """
        This function parses the given initial input files and obtains the execution label ("BUG"/"NO_BUG")
        :return:
        """
        for inp in self._initial_inputs:
            try:
                self._inputs.add(
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

        for inp in self._inputs:
            inp.oracle = self._execute_input(inp)

        num_bug_inputs = len([inp for inp in self._inputs if inp.oracle == OracleResult.BUG])

        if num_bug_inputs < self._targeted_start_size:
            self._inputs.update(
                generate_inputs(
                    self._inputs,
                    grammar=self._grammar,
                    prop=self._prop,
                    max_positive_samples=self._targeted_start_size,
                    max_negative_samples=self._targeted_start_size
                )
            )


    @time
    def execute(self):
        logging.info("Starting AVICENNA.")
        register_termination(self._timeout)
        try:
            self._start_time = perf_counter()
            self._setup()
            while self._do_more_iterations():
                logging.info("Starting Iteration " + str(self._iteration))
                new_inputs = self._loop(self._inputs)
                self._inputs.update(new_inputs)
                self._iteration = self._iteration + 1
        except CustomTimeout:
            logging.exception("Terminate due to timeout")
        return self._finalize()

    def _do_more_iterations(self):
        if self._iteration >= self._max_iterations:
            logging.info("Terminate due to maximal iterations reached")
            return False
        return True

    def _loop(self, test_inputs):
        # Combining with the already existing data
        # self._add_new_data(new_inputs, execution_outcome)
        assert all(map(lambda x: isinstance(x.oracle, OracleResult), self._inputs))

        # Extract the most important non-terminals that are responsible for the program behavior
        excluded_non_terminals = self._get_exclusion_set(test_inputs)

        # Run islearn to learn new constraints
        new_constraints = self._learn_new_constraints(test_inputs, excluded_non_terminals)

        # Evaluate Invariants
        current_best_invariants = self._evaluate_constraints(test_inputs, new_constraints)

        # Compare to previous invariants to get the best global invariant
        self._best_candidates = self._get_best_global_invariant(current_best_invariants)

        # best_invariants = self._get_best_invariants(new_constraints)
        negated_constraints = self._negating_constraints(list(self._best_candidates))

        # Generate new input samples form the learned constraints
        new_inputs = self._generate_new_inputs(
            negated_constraints + list(self._best_candidates)
        )

        for inp in new_inputs:
            inp.oracle = self._execute_input(inp)

        return new_inputs

    @time
    def _learn_new_constraints(self, test_inputs: Set[Input], excluded_non_terminals: Set[str]) -> List[str]:
        logging.info("Learning new failure-constraints.")
        positive_trees = []
        negative_trees = []
        for inp in test_inputs:
            if inp.oracle == OracleResult.BUG:
                positive_trees.append(inp.tree)
            else:
                negative_trees.append(inp.tree)

        assert len(positive_trees) != 0 and len(negative_trees) != 0

        new_constraints = list()
        try:
            new_constraints = run_islearn(
                grammar=self._grammar,
                prop=self._prop,
                positive_trees=positive_trees,
                negative_trees=negative_trees,
                activated_patterns=self._activated_patterns,
                excluded_features=excluded_non_terminals,
                pattern_file=self._pattern_file,
                # max_conjunction_size=self._max_conjunction_size,
            )
        except ValueError as e:
            logging.info(e)
            logging.info(f"Could not learn any constraints")

        if len(new_constraints) == 0:
            logging.info(f"Could not learn any constraints")
            if self._best_candidates:
                new_constraints = self._best_candidates
            else:
                logging.info(f"Retrying with relaxed conditions")
                new_constraints = run_islearn(
                    grammar=self._grammar,
                    prop=self._prop,
                    positive_trees=positive_trees,
                    negative_trees=negative_trees,
                    activated_patterns=self._activated_patterns,
                    pattern_file=self._pattern_file,
                )
                print(new_constraints)

        assert len(new_constraints) != 0, "No new candidate constraints were learned. Exiting."
        if len(new_constraints) == 0:
            new_constraints = self._learn_new_constraints(set())

        for i in new_constraints:
            logging.info(i)

        return new_constraints

    @time
    def _get_exclusion_set(
            self, test_inputs: Set[Input]
    ) -> Set[str]:
        logging.info("Determining the most important non-terminals.")

        learner = InputElementLearner(
            grammar=self._grammar,
            prop=self._prop,
            input_samples=test_inputs,
            max_relevant_features=self._max_excluded_features,
        )
        learner.learn()
        relevant, irrelevant = learner.get_exclusion_sets()

        logging.info(f"Determined: {relevant} to be the most relevant features.")
        logging.info(f"Excluding: {irrelevant} from candidate consideration.")

        return irrelevant

    def _negating_constraints(self, constraints: List):
        negated_constraints = []
        for constraint in constraints:
            negated_constraints.append("not(" + constraint + ")")

        return negated_constraints

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
                    logging.info("Removing constraint because no new inputs were generated")
                    # Solver was not able to generate inputs:
                    self._add_infeasible_constraints(constraint)
                else:
                    inputs = inputs + new_input_trees

        assert len(inputs) != 0, "No new inputs were generated. Exiting."
        logging.info(f"{len(inputs)} new inputs have been generated.")

        new_inputs = set()
        for tree in inputs:
            new_inputs.add(
                Input(
                    tree=tree
                )
            )
        return new_inputs

    def _add_infeasible_constraints(self, constraint):
        self._infeasible_constraints.add(constraint)
        logging.info("Removing infeasible constraint")
        logging.debug(f"Infeasible constraint: {constraint}")

    def _get_best_invariants(self, new_constraints):
        # constraints = list(map(lambda p: f"" + ISLaUnparser(p[0]).unparse(), new_constraints.items()))

        logging.info(f"In total, {len(new_constraints)} constraints were learned.\n")
        logging.info(
            "\n".join(
                map(
                    lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse() + "\n",
                    new_constraints.items(),
                )
            )
        )

        best_invariant, (specificity, sensitivity) = next(iter(new_constraints.items()))
        logging.info(
            f"Best invariant (*estimated* specificity {specificity:.2f}, sensitivity: {sensitivity:.2f}):"
        )
        logging.info(ISLaUnparser(best_invariant).unparse())

        self._learned_invariants[best_invariant] = [specificity, sensitivity]

        return best_invariant

    def _finalize(self):
        logging.info("Avicenna finished")
        logging.info("The best learned failure invariant(s):")

        best = dict()
        best_f1 = 0
        for constraint in self._best_candidates.items():
            p0, p1 = constraint
            if p1[2] >= best_f1:
                best_f1 = p1[2]
                best[p0] = p1

        logging.info("\n".join(map(lambda p: f"{p[1]}: " + p[0] + "\n", best.items())))

        return best

    def _add_new_data(self, inputs: List[DerivationTree], exec_oracle: List[bool]):
        df = pandas.DataFrame(
            list(zip(inputs, exec_oracle)), columns=["input", "oracle"]
        )

        if self._all_data is None:
            self._all_data = df.drop_duplicates()
        else:
            self._all_data = pandas.concat(
                [self._all_data, df], sort=False
            ).drop_duplicates()

    @time
    def _evaluate_constraints(self, test_inputs: Set[Input], constraints):
        logging.info(f"Evaluating {len(constraints)} constraints.")
        eval_data: Dict[str, List[float]] = {}

        for constraint in constraints:
            constraint_data: Dict = constraint_eval(
                test_inputs,
                constraint,
                grammar=self._grammar,
            )

            precision = precision_score(
                constraint_data["oracle"].astype(bool),  # TODO Reformat
                constraint_data["predicted"].astype(bool),
                pos_label=True,
                average="binary",
            )
            precision = round(precision * 100, 3)

            recall = recall_score(
                constraint_data["oracle"].astype(bool),
                constraint_data["predicted"].astype(bool),
                pos_label=True,
                average="binary",
            )
            recall = round(recall * 100, 3)

            f1 = f1_score(
                constraint_data["oracle"].astype(bool),
                constraint_data["predicted"].astype(bool),
                pos_label=True,
                average="binary",
            )

            logging.debug(f"Results for f{constraint}")
            logging.debug(f"The constraint achieved a precision of {precision} %, recall of {recall} % and f1-score of {round(f1, 3)}")

            eval_data[constraint] = [precision, recall, f1]

        sorted_data = sorted(
            eval_data.items(), key=lambda item: item[1][2], reverse=True
        )

        sorted_comp = {}
        for s in sorted_data:
            sorted_comp[s[0]] = s[1]

        best_five = list(sorted_comp.keys())[0:4]  # TODO Abstract
        # for i in best_five:
        #    print(i)

        # self._best_invariant = list(sorted_comp.keys())[0]  # TODO Select the global best Invariant
        return sorted_comp

    def _get_best_global_invariant(self, current_best_invariants):

        if self._iteration != 0:
            combined_data = self._best_candidates | current_best_invariants
        else:
            combined_data = current_best_invariants

        sorted_data = sorted(
            combined_data.items(), key=lambda item: item[1][2], reverse=True
        )
        best_global_constraints = {}
        for i in list(sorted_data)[0:4]:
            best_global_constraints[i[0]] = combined_data[i[0]]

        logging.info(f"Top five learned constraints:\n")
        logging.info(
            "\n".join(
                map(
                    lambda p: f"{p[1]}: " + p[0] + "\n", best_global_constraints.items()
                )
            )
        )

        return best_global_constraints

    def iteration_identifier_map(self):
        return {"iteration": self._iteration}
