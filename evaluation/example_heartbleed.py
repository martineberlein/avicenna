import string
import math

from fuzzingbook.Grammars import Grammar
from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna_formalizations.heartbeat import grammar, oracle, initial_inputs, oracle_simple



if __name__ == "__main__":
    # Length Feature needs to be activated

    avicenna = Avicenna(
        grammar=grammar,
        initial_inputs=initial_inputs,
        oracle=oracle,
        max_iterations=5,
    )

    diagnoses = avicenna.explain()

    for diagnosis in diagnoses:
        print(ISLaUnparser(diagnosis[0]).unparse())
