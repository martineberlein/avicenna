import logging
from typing import List, Tuple, Set, Callable

from fuzzingbook.Parser import EarleyParser

from islearn.mutation import MutationFuzzer
from islearn.helpers import tree_in
from isla.fuzzer import GrammarCoverageFuzzer
from isla.language import DerivationTree

from fuzzingbook.Grammars import Grammar

from avicenna.input import Input
from avicenna.oracle import OracleResult


def generate_inputs(
    test_inputs: Set[Input],
    grammar: Grammar,
    prop: Callable[[Input], OracleResult],
    max_positive_samples: int = 100,
    max_negative_samples: int = 100,
) -> Set[Input]:
    logging.info(f"Generating more inputs.")
    new_inputs = set()

    positive_trees = []
    negative_trees = []
    for inp in test_inputs:
        if inp.oracle == OracleResult.BUG:
            positive_trees.append(inp.tree)
        else:
            negative_trees.append(inp.tree)

    generator = Generator(
        max_positive=max_positive_samples,
        max_negative=max_negative_samples,
        grammar=grammar,
        prop=prop,
    )
    pos_inputs, neg_inputs = generator.generate_mutation(positive_trees)
    for tree in pos_inputs + neg_inputs:
        new_inputs.add(Input(tree=tree))
    # Can be improved by directly creating new Inputs with BUG or NO_BUG
    for inp in new_inputs:
        label = prop(inp)
        inp.oracle = OracleResult.BUG if label else OracleResult.NO_BUG

    return new_inputs


class Generator:
    def __init__(self, max_positive, max_negative, grammar: Grammar, prop):
        self._max_positive_samples = max_positive
        self._max_negative_samples = max_negative
        self._grammar = grammar
        self._prop = prop

    def generate_mutation(
        self, positive_samples: List[DerivationTree | str]
    ) -> Tuple[List[DerivationTree], List[DerivationTree]]:
        if not isinstance(positive_samples[0], DerivationTree):
            logging.debug("Transforming str-inputs to derivation trees.")
            positive_trees = [
                DerivationTree.from_parse_tree(
                    next(EarleyParser(self._grammar).parse(inp))
                )
                for inp in positive_samples
            ]
        else:
            positive_trees = positive_samples

        validation_inputs = positive_trees
        negative_validation_inputs = []

        # We run two mutation fuzzers and a grammar fuzzer in parallel
        mutation_fuzzer = MutationFuzzer(self._grammar, positive_trees, self._prop, k=3)
        mutate_fuzz = mutation_fuzzer.run(900, alpha=0.1, yield_negative=True)

        grammar_fuzzer = GrammarCoverageFuzzer(self._grammar)

        i = 0
        while (
            len(validation_inputs) < self._max_positive_samples
            or len(negative_validation_inputs) < self._max_negative_samples
        ):
            if i % 10 == 0:
                logging.debug(
                    f"Fuzzing: {len(validation_inputs):02} positive / {len(negative_validation_inputs):02} negative "
                    f"inputs"
                )

            fuzzer_inputs = [
                next(mutate_fuzz),
                grammar_fuzzer.expand_tree(DerivationTree("<start>", None)),
            ]

            for idx, inp in enumerate(fuzzer_inputs):
                if (
                    len(validation_inputs) < self._max_positive_samples
                    and self._prop(inp)
                    and not tree_in(inp, validation_inputs)
                ):
                    validation_inputs.append(inp)
                    if idx == 0:
                        mutation_fuzzer.population.add(inp)
                elif (
                    len(negative_validation_inputs) < self._max_negative_samples
                    and not self._prop(inp)
                    and not tree_in(inp, negative_validation_inputs)
                ):
                    negative_validation_inputs.append(inp)

            i += 1

        return validation_inputs, negative_validation_inputs
