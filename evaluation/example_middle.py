from enum import Enum
import logging

from fuzzingbook.Grammars import Grammar, is_valid_grammar
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna import Avicenna
from isla.language import ISLaUnparser

from avicenna_formalizations.middle import grammar, oracle, initial_inputs


if __name__ == "__main__":
    avicenna = Avicenna(
        grammar=grammar,
        initial_inputs=["3, 1, 4", "3, 2, 1"],
        oracle=oracle,
        max_excluded_features=4,
        max_iterations=10,
    )

    diagnoses = avicenna.explain()

    for diagnosis in diagnoses:
        print(ISLaUnparser(diagnosis[0]).unparse())
