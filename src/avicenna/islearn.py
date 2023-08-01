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
from islearn.learner import InvariantLearner
from islearn.learner import patterns_from_file

STANDARD_PATTERNS_REPO = "patterns.toml"
logger = logging.getLogger("learner")

from avicenna.input import Input
from avicenna.oracle import OracleResult


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
        self.exclude_nonterminals = set([])
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

    def learn_failure_invariants(
        self,
        test_inputs: Optional[Iterable[Input]] = None,
        exclude_nonterminals: Optional[Iterable[str]] = None,
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

        self.exclude_nonterminals = exclude_nonterminals or set([])

        return self.learn_invariants()
