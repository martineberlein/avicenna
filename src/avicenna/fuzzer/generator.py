import logging
from typing import List, Tuple

from fuzzingbook.Parser import EarleyParser

from islearn.mutation import MutationFuzzer
from islearn.helpers import tree_in
from isla.fuzzer import GrammarCoverageFuzzer
from isla.language import DerivationTree

from fuzzingbook.Grammars import Grammar


class Generator:
    def __init__(self, max_positive, max_negative, grammar: Grammar, prop):
        self._max_positive_samples = max_positive
        self._max_negative_samples = max_negative
        self._grammar = grammar
        self._prop = prop

    def generate_mutation(
        self, positive_samples: List[DerivationTree | str], negative_samples: List[DerivationTree | str]
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
        negative_validation_inputs = negative_samples  # TODO Handle None if there are no neg_samples

        def reverse_prop(inp):
            o = self._prop(inp)
            if isinstance(o, bool):
                return not o
            else:
                return -1

        # We run two mutation fuzzers and a grammar fuzzer in parallel
        mutation_fuzzer = MutationFuzzer(self._grammar, positive_trees, self._prop, k=3)
        mutation_fuzzer_negative = MutationFuzzer(self._grammar, negative_validation_inputs, reverse_prop, k=3)
        mutate_fuzz = mutation_fuzzer.run(900, alpha=0.1, yield_negative=True)
        mutate_fuzz_negative = mutation_fuzzer_negative.run(900, alpha=0.1, yield_negative=True)

        grammar_fuzzer = GrammarCoverageFuzzer(self._grammar)

        i = 0
        while (
            len(validation_inputs) < self._max_positive_samples
            or len(negative_validation_inputs) < self._max_negative_samples
        ):
            if i % 10 == 0:
                logging.info(
                    f"Fuzzing: {len(validation_inputs):02} positive / {len(negative_validation_inputs):02} negative "
                    f"inputs"
                )

            fuzzer_inputs = [
                next(mutate_fuzz),
                next(mutate_fuzz_negative),
                # grammar_fuzzer.expand_tree(DerivationTree("<start>", None)),
            ]

            for idx, inp in enumerate(fuzzer_inputs):
                oracle = self._prop(inp)
                if oracle == -1:
                    pass
                elif (
                    len(validation_inputs) < self._max_positive_samples
                    and oracle
                    and not tree_in(inp, validation_inputs)
                ):
                    validation_inputs.append(inp)
                    if idx == 0:
                        mutation_fuzzer.population.add(inp)
                elif (
                    len(negative_validation_inputs) < self._max_negative_samples
                    and not oracle
                    and not tree_in(inp, negative_validation_inputs)
                ):
                    negative_validation_inputs.append(inp)

            i += 1

        return validation_inputs, negative_validation_inputs