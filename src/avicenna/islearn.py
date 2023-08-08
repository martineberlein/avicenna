import copy
import functools
import itertools
from typing import List, Tuple, Set, Dict, Optional, cast, Callable, Iterable, Sequence

import isla.fuzzer
from isla import language, isla_predicates
from isla.helpers import RE_NONTERMINAL
from isla.language import ensure_unique_bound_variables, Formula
from islearn.reducer import InputReducer


import inspect
import logging
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Callable, Iterable

import isla.fuzzer
from grammar_graph import gg
from isla import language, isla_predicates
from isla.helpers import (
    RE_NONTERMINAL,
    canonical,
)
from isla.type_defs import Grammar
from isla.z3_helpers import evaluate_z3_expression
from islearn.language import parse_abstract_isla
from islearn.learner import InvariantLearner, TruthTable, TruthTableRow
from islearn.learner import patterns_from_file
from islearn.learner import InvariantLearner

STANDARD_PATTERNS_REPO = "patterns.toml"
logger = logging.getLogger("learner")

from avicenna.input import Input
from avicenna.oracle import OracleResult
from avicenna.result_table import TruthTable, TruthTableRow


class AviIslearn(InvariantLearner):
    def __init__(
        self,
        grammar: Grammar,
        prop: Optional[Callable[[language.DerivationTree], OracleResult]] = None,
        patterns: Optional[List[language.Formula | str]] = None,
        pattern_file: Optional[str] = None,
        activated_patterns: Optional[Iterable[str]] = None,
        deactivated_patterns: Optional[Iterable[str]] = None,
    ):
        super().__init__(grammar)
        self.grammar = grammar
        self.canonical_grammar = canonical(grammar)
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.prop = prop

        assert not activated_patterns or not deactivated_patterns
        self.patterns = []
        if not patterns:
            pattern_repo = patterns_from_file(pattern_file)
            if activated_patterns:
                self.patterns = [
                    pattern
                    for name in activated_patterns
                    for pattern in pattern_repo[name]
                ]
            else:
                self.patterns = list(
                    pattern_repo.get_all(but=deactivated_patterns or [])
                )
        else:
            self.patterns = [
                pattern
                if isinstance(pattern, language.Formula)
                else parse_abstract_isla(pattern, grammar)
                for pattern in patterns
            ]

        # Set later
        self.exclude_non_terminals: Set[str] = set([])
        self.positive_examples: Set[Input] = set()
        self.negative_examples: Set[Input] = set()
        self.positive_examples_for_learning: List[language.DerivationTree] = []

    def learn_failure_invariants(
        self,
        test_inputs: Set[Input],
        precision_truth_table: TruthTable,
        recall_truth_table: TruthTable,
        exclude_non_terminals: Optional[Iterable[str]] = None,
    ) -> Dict[language.Formula, Tuple[float, float]]:
        positive_inputs = set(
            [inp.tree for inp in test_inputs if inp.oracle == OracleResult.BUG]
        ) or set([])
        negative_inputs = set(
            [inp.tree for inp in test_inputs if inp.oracle == OracleResult.NO_BUG]
        ) or set([])

        self.exclude_non_terminals = exclude_non_terminals or set([])

        self._learn_invariants(positive_inputs, negative_inputs, recall_truth_table)

        return dict()

    def _learn_invariants(
        self, positive_inputs: Set[Input], negative_inputs: Set[Input], precision_truth_table:TruthTable, recall_truth_table: TruthTable, max_number_positive_inputs_for_learning=10
    ):
        positive_inputs_for_learning = self._sort_inputs(
            self.original_positive_examples,
            self.filter_inputs_for_learning_by_kpaths,
            more_paths_weight=1.7,
            smaller_inputs_weight=1.0,
        )[:max_number_positive_inputs_for_learning]

        candidates = self.generate_candidates(
            self.patterns, positive_inputs_for_learning
        )

        logger.info("Found %d invariant candidates.", len(candidates))

        recall_truth_table.evaluate(positive_inputs, self.graph)

        # Update new Candidates
        for candidate in candidates:
            if hash(candidate) not in recall_truth_table.row_hashes:
                self.truthTable.append(
                    TruthTableRow(candidate).evaluate(self.all_inputs, self._graph)
                )

        # Deleting throws away all calculated evals so far == bad -> maybe only pass TruthTableRows >= self.min_recall?
        if self.max_disjunction_size < 2:
            for row in recall_truth_table:
                if row.eval_result() < self.min_recall:
                    recall_truth_table.remove(row)



        pass

    def reduce_inputs(
        self, test_inputs: Set[Input], negative_inputs: Set[Input]
    ) -> Tuple[Set[Input], Set[Input]]:
        raise NotImplementedError()


class AvicennaISlearn(InvariantLearner):
    def __init__(
        self,
        grammar: Grammar,
        prop: Optional[Callable[[language.DerivationTree], bool]] = None,
        patterns: Optional[List[language.Formula | str]] = None,
        pattern_file: Optional[str] = None,
        activated_patterns: Optional[Iterable[str]] = None,
        deactivated_patterns: Optional[Iterable[str]] = None,
        k: int = 3,
        target_number_positive_samples: int = 10,
        target_number_negative_samples: int = 10,
        target_number_positive_samples_for_learning: int = 10,
        mexpr_expansion_limit: int = 1,
        max_nonterminals_in_mexpr: Optional[int] = None,
        min_recall: float = 0.9,
        min_specificity: float = 0.6,
        max_disjunction_size: int = 1,
        max_conjunction_size: int = 2,
        include_negations_in_disjunctions: bool = False,
        reduce_inputs_for_learning: bool = True,
        reduce_all_inputs: bool = False,
        generate_new_learning_samples: bool = True,
        do_generate_more_inputs: bool = True,
        filter_inputs_for_learning_by_kpaths: bool = True,
    ):
        # We add extended caching certain, crucial functions.
        super().__init__(
            grammar,
            prop,
            None,
            None,
            patterns,
            None,
            activated_patterns,
            deactivated_patterns,
            k,
            target_number_positive_samples,
            target_number_negative_samples,
            target_number_positive_samples_for_learning,
            mexpr_expansion_limit,
            max_nonterminals_in_mexpr,
            min_recall,
            min_specificity,
            max_disjunction_size,
            max_conjunction_size,
            None,
            include_negations_in_disjunctions,
            reduce_inputs_for_learning,
            reduce_all_inputs,
            generate_new_learning_samples,
            do_generate_more_inputs,
            filter_inputs_for_learning_by_kpaths,
        )
        isla.helpers.evaluate_z3_expression = lru_cache(maxsize=None)(
            inspect.unwrap(evaluate_z3_expression)
        )
        isla.language.DerivationTree.__str__ = lru_cache(maxsize=None)(
            inspect.unwrap(isla.language.DerivationTree.__str__)
        )
        isla.language.DerivationTree.paths = lru_cache(maxsize=128)(
            inspect.unwrap(isla.language.DerivationTree.paths)
        )
        isla.language.DerivationTree.__hash__ = lambda tree: tree.id
        isla.isla_predicates.is_nth = lru_cache(maxsize=128)(
            inspect.unwrap(isla.isla_predicates.is_nth)
        )

        self.grammar = grammar
        self.canonical_grammar = canonical(grammar)
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.prop = prop
        self.k = k
        self.mexpr_expansion_limit = mexpr_expansion_limit
        self.max_nonterminals_in_mexpr = max_nonterminals_in_mexpr
        self.min_recall = min_recall
        self.min_specificity = min_specificity
        self.max_disjunction_size = max_disjunction_size
        self.max_conjunction_size = max_conjunction_size
        self.include_negations_in_disjunctions = include_negations_in_disjunctions
        self.reduce_inputs_for_learning = reduce_inputs_for_learning
        self.reduce_all_inputs = reduce_all_inputs
        self.generate_new_learning_samples = generate_new_learning_samples
        self.do_generate_more_inputs = do_generate_more_inputs
        self.filter_inputs_for_learning_by_kpaths = filter_inputs_for_learning_by_kpaths

        # Set later
        self.exclude_non_terminals = set([])
        self.positive_examples: List[language.DerivationTree] = list()
        self.original_positive_examples: List[language.DerivationTree] = list(
            self.positive_examples
        )
        self.negative_examples: List[language.DerivationTree] = list()
        self.positive_examples_for_learning: List[language.DerivationTree] = []

        self.target_number_positive_samples = target_number_positive_samples
        self.target_number_negative_samples = target_number_negative_samples
        self.target_number_positive_samples_for_learning = (
            target_number_positive_samples_for_learning
        )
        assert (
            target_number_positive_samples
            >= target_number_positive_samples_for_learning
        )

        assert not prop or all(prop(example) for example in self.positive_examples)
        assert not prop or all(not prop(example) for example in self.negative_examples)

        # Also consider inverted patterns?
        assert not activated_patterns or not deactivated_patterns
        self.patterns = []
        if not patterns:
            pattern_repo = patterns_from_file(pattern_file)
            if activated_patterns:
                self.patterns = [
                    pattern
                    for name in activated_patterns
                    for pattern in pattern_repo[name]
                ]
            else:
                self.patterns = list(
                    pattern_repo.get_all(but=deactivated_patterns or [])
                )
        else:
            self.patterns = [
                pattern
                if isinstance(pattern, language.Formula)
                else parse_abstract_isla(pattern, grammar)
                for pattern in patterns
            ]

        self.previously_seen_invariants: Set[language.Formula] = set()
        self.precision_truth_table = TruthTable()
        self.recall_truth_table = TruthTable()

    def learn_failure_invariants(
        self,
        test_inputs: Optional[Iterable[Input]] = None,
        exclude_non_terminals: Optional[Iterable[str]] = None,
    ) -> Dict[language.Formula, Tuple[float, float]]:
        self.positive_examples = [
            inp.tree for inp in test_inputs if inp.oracle == OracleResult.BUG
        ] or []
        self.negative_examples = [
            inp.tree for inp in test_inputs if inp.oracle == OracleResult.NO_BUG
        ] or []

        self.original_positive_examples: List[language.DerivationTree] = list(
            self.positive_examples
        )

        assert not self.prop or all(
            self.prop(example) for example in self.positive_examples
        )
        assert not self.prop or all(
            not self.prop(example) for example in self.negative_examples
        )

        self.exclude_non_terminals = exclude_non_terminals or set([])

        return self.learn_invariants()

    def learn_invariants(
        self, ensure_unique_var_names: bool = True
    ) -> Dict[language.Formula, Tuple[float, float]]:
        if self.prop and self.do_generate_more_inputs:
            self._generate_more_inputs()
            assert (
                len(self.positive_examples) > 0
            ), "Cannot learn without any positive examples!"
            assert all(
                self.prop(positive_example)
                for positive_example in self.positive_examples
            )
            assert all(
                not self.prop(negative_example)
                for negative_example in self.negative_examples
            )

        if self.reduce_all_inputs and self.prop is not None:
            logger.info(
                "Reducing %d positive samples w.r.t. property and k=%d.",
                len(self.positive_examples),
                self.k,
            )
            reducer = InputReducer(self.grammar, self.prop, self.k)
            self.positive_examples = [
                reducer.reduce_by_smallest_subtree_replacement(inp)
                for inp in self.positive_examples
            ]

            logger.info(
                "Reducing %d negative samples w.r.t. property and k=%d.",
                len(self.negative_examples),
                self.k,
            )
            reducer = InputReducer(self.grammar, lambda t: not self.prop(t), self.k)
            self.negative_examples = [
                reducer.reduce_by_smallest_subtree_replacement(inp)
                for inp in self.negative_examples
            ]

        if self.generate_new_learning_samples or not self.original_positive_examples:
            self.positive_examples_for_learning = self._sort_inputs(
                self.positive_examples,
                self.filter_inputs_for_learning_by_kpaths,
                more_paths_weight=1.7,
                smaller_inputs_weight=1.0,
            )[: self.target_number_positive_samples_for_learning]
        else:
            self.positive_examples_for_learning = self._sort_inputs(
                self.original_positive_examples,
                self.filter_inputs_for_learning_by_kpaths,
                more_paths_weight=1.7,
                smaller_inputs_weight=1.0,
            )[: self.target_number_positive_samples_for_learning]

        logger.info(
            "Keeping %d positive examples for candidate generation.",
            len(self.positive_examples_for_learning),
        )

        if (
            (not self.reduce_all_inputs or not self.generate_new_learning_samples)
            and self.reduce_inputs_for_learning
            and self.prop is not None
        ):
            logger.info(
                "Reducing %d inputs for learning w.r.t. property and k=%d.",
                len(self.positive_examples_for_learning),
                self.k,
            )
            reducer = InputReducer(self.grammar, self.prop, self.k)
            self.positive_examples_for_learning = [
                reducer.reduce_by_smallest_subtree_replacement(inp)
                for inp in self.positive_examples_for_learning
            ]

        logger.debug(
            "Examples for learning:\n%s",
            "\n".join(map(str, self.positive_examples_for_learning)),
        )

        candidates = self.generate_candidates(
            self.patterns, self.positive_examples_for_learning
        )
        logger.info("Found %d invariant candidates.", len(candidates))

        logger.debug(
            "Candidates:\n%s",
            "\n\n".join(
                [language.ISLaUnparser(candidate).unparse() for candidate in candidates]
            ),
        )

        logger.info("Filtering invariants.")

        # Only consider *real* invariants

        # NOTE: Disabled parallel evaluation for now. In certain cases, this renders
        #       the filtering process *much* slower, or gives rise to stack overflows
        #       (e.g., "test_learn_from_islearn_patterns_file" example).
        recall_truth_table = TruthTable(
            [TruthTableRow(inv, self.positive_examples) for inv in candidates]
        ).evaluate(
            self.graph,
            # rows_parallel=True,
            lazy=self.max_disjunction_size < 2,
            result_threshold=self.min_recall,
        )

        if self.max_disjunction_size < 2:
            for row in recall_truth_table:
                if row.eval_result() < self.min_recall:
                    recall_truth_table.remove(row)

        precision_truth_table = None
        if self.negative_examples:
            logger.info("Evaluating precision.")
            logger.debug(
                "Negative samples:\n"
                + "\n-----------\n".join(map(str, self.negative_examples))
            )

            # TODO
            # Speed Improvement -> Limiting factor == all negative samples get evaluated all the time
            #           But: Many invariants are learned more than once during each iteration of avicenna
            #
            # Solution: Have a single precision_truth_table (maybe also recall_truth_table)
            #           Only add new rows if not included yet -> complete eval with all inputs
            #           For existing: only eval newly added inputs
            #
            #         For subsequence invariant learning, only use those candidates specified in the recall_truth_table

            precision_truth_table = TruthTable(
                [
                    TruthTableRow(row.formula, self.negative_examples)
                    for row in recall_truth_table
                ]
            ).evaluate(
                self.graph,
                # rows_parallel=True
            )

            assert len(precision_truth_table) == len(recall_truth_table)

        assert not self.negative_examples or precision_truth_table is not None

        invariants = {
            row.formula
            for row in recall_truth_table
            if row.eval_result() >= self.min_recall
        }

        logger.info(
            "%d invariants with recall >= %d%% remain after filtering.",
            len(invariants),
            int(self.min_recall * 100),
        )

        logger.debug(
            "Invariants:\n%s",
            "\n\n".join(map(lambda f: language.ISLaUnparser(f).unparse(), invariants)),
        )

        if self.max_disjunction_size > 1:
            logger.info("Calculating recall of Boolean combinations.")

            disjunctive_recall_truthtable = copy.deepcopy(recall_truth_table)
            assert precision_truth_table is None or len(
                disjunctive_recall_truthtable
            ) == len(precision_truth_table)

            for level in range(2, self.max_disjunction_size + 1):
                assert precision_truth_table is None or len(
                    disjunctive_recall_truthtable
                ) == len(precision_truth_table)
                logger.debug(f"Disjunction size: {level}")

                for rows_with_indices in itertools.combinations(
                    enumerate(recall_truth_table), level
                ):
                    assert precision_truth_table is None or len(
                        disjunctive_recall_truthtable
                    ) == len(precision_truth_table)
                    assert precision_truth_table is None or all(
                        rwi[1].formula == precision_truth_table[rwi[0]].formula
                        for rwi in rows_with_indices
                    )

                    max_num_negations = (
                        level // 2 if self.include_negations_in_disjunctions else 0
                    )
                    for formulas_to_negate in (
                        t
                        for t in itertools.product(*[[0, 1] for _ in range(3)])
                        if sum(t) <= max_num_negations
                    ):
                        # To ensure that only "meaningful" properties are negated, the un-negated properties
                        # should hold for at least 20% of all inputs; but at most for 80%, since otherwise,
                        # the negation is overly specific.
                        negated_rows: List[Tuple[int, TruthTableRow]] = list(
                            itertools.compress(rows_with_indices, formulas_to_negate)
                        )
                        if any(
                            0.8 < negated_row.eval_result() < 0.2
                            for _, negated_row in negated_rows
                        ):
                            continue

                        recall_table_rows = [
                            -row if bool(negate) else row
                            for negate, (_, row) in zip(
                                formulas_to_negate, rows_with_indices
                            )
                        ]

                        # Compute recall of disjunction, add if above threshold and
                        # an improvement over all participants of the disjunction
                        disjunction = functools.reduce(
                            TruthTableRow.__or__, recall_table_rows
                        )
                        new_recall = disjunction.eval_result()
                        if new_recall < self.min_recall or not all(
                            new_recall > row.eval_result() for row in recall_table_rows
                        ):
                            continue

                        disjunctive_recall_truthtable.append(disjunction)

                        if precision_truth_table is not None:
                            # Also add disjunction to the precision truth table. Saves us a couple of evaluations.
                            precision_table_rows = [
                                -precision_truth_table[idx]
                                if bool(negate)
                                else precision_truth_table[idx]
                                for negate, (idx, _) in zip(
                                    formulas_to_negate, rows_with_indices
                                )
                            ]
                            disjunction = functools.reduce(
                                TruthTableRow.__or__, precision_table_rows
                            )
                            precision_truth_table.append(disjunction)

            recall_truth_table = disjunctive_recall_truthtable

            invariants = {
                row.formula
                for row in recall_truth_table
                if row.eval_result() >= self.min_recall
            }

            logger.info(
                "%d invariants with recall >= %d%% remain after building Boolean combinations.",
                len(invariants),
                int(self.min_recall * 100),
            )

            # logger.debug(
            #   "Invariants:\n%s", "\n\n".join(map(lambda f: language.ISLaUnparser(f).unparse(), invariants)))

        # assert all(evaluate(row.formula, inp, self.grammar).is_true()
        #            for inp in self.positive_examples
        #            for row in recall_truth_table
        #            if row.eval_result() == 1)

        if not self.negative_examples:
            # TODO: Enforce unique names, sort
            if ensure_unique_var_names:
                invariants = sorted(
                    list(map(ensure_unique_bound_variables, invariants)),
                    key=lambda inv: len(inv),
                )
            else:
                invariants = sorted(list(invariants), key=lambda inv: len(inv))
            return {
                row.formula: (1.0, row.eval_result())
                for row in recall_truth_table
                if row.eval_result() >= self.min_recall
            }

        indices_to_remove = list(
            reversed(
                [
                    idx
                    for idx, row in enumerate(recall_truth_table)
                    if row.eval_result() < self.min_recall
                ]
            )
        )
        assert sorted(indices_to_remove, reverse=True) == indices_to_remove
        for idx in indices_to_remove:
            recall_truth_table.remove(recall_truth_table[idx])
            precision_truth_table.remove(precision_truth_table[idx])

        logger.info("Calculating precision of Boolean combinations.")
        conjunctive_precision_truthtable = copy.copy(precision_truth_table)
        for level in range(2, self.max_conjunction_size + 1):
            logger.debug(f"Conjunction size: {level}")
            assert len(recall_truth_table) == len(conjunctive_precision_truthtable)
            for rows_with_indices in itertools.combinations(
                enumerate(precision_truth_table), level
            ):
                precision_table_rows = [row for (_, row) in rows_with_indices]

                # Only consider combinations where all rows meet minimum recall requirement.
                # Recall doesn't get better by forming conjunctions!
                if any(
                    recall_truth_table[idx].eval_result() < self.min_recall
                    for idx, _ in rows_with_indices
                ):
                    continue

                # Compute precision of conjunction, add if above threshold and
                # an improvement over all participants of the conjunction
                conjunction = functools.reduce(
                    TruthTableRow.__and__, precision_table_rows
                )
                # if conjunction.formula in self.previously_seen_invariants:
                #    print("--------------------")
                #    continue
                new_precision = 1 - conjunction.eval_result()
                if new_precision < self.min_specificity or not all(
                    new_precision > 1 - row.eval_result()
                    for row in precision_table_rows
                ):
                    continue

                conjunctive_precision_truthtable.append(conjunction)

                recall_table_rows = [
                    recall_truth_table[idx] for idx, _ in rows_with_indices
                ]
                conjunction = functools.reduce(TruthTableRow.__and__, recall_table_rows)
                recall_truth_table.append(conjunction)

        precision_truth_table = conjunctive_precision_truthtable

        # assert all(evaluate(row.formula, inp, self.grammar).is_false()
        #            for inp in self.negative_examples
        #            for row in precision_truth_table
        #            if row.eval_result() == 0)

        result: Dict[language.Formula, Tuple[float, float]] = {
            precision_row.formula
            if not ensure_unique_var_names
            else language.ensure_unique_bound_variables(precision_row.formula): (
                1 - precision_row.eval_result(),
                recall_truth_table[idx].eval_result(),
            )
            for idx, precision_row in enumerate(precision_truth_table)
            if (
                1 - precision_row.eval_result() >= self.min_specificity
                and recall_truth_table[idx].eval_result() >= self.min_recall
            )
        }

        logger.info(
            "Found %d invariants with precision >= %d%%.",
            len([p for p in result.values() if p[0] >= self.min_specificity]),
            int(self.min_specificity * 100),
        )

        # TODO: Sort within same recall / specificity values: Fewer disjunctions,
        #       more common String constants... To optimize specificity further.

        # for cond in result.keys():
        #     if cond in self.previously_seen_invariants:
        #        print("FOUND ONE!")
        #        print(language.ISLaUnparser(cond).unparse(), "\n\n")

        # for cond in result.keys():
        #    self.previously_seen_invariants.add(cond)

        return dict(
            cast(
                List[Tuple[language.Formula, Tuple[float, float]]],
                sorted(result.items(), key=lambda p: (p[1], -len(p[0])), reverse=True),
            )
        )
