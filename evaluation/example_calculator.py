from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna_formalizations.calculator import grammar, oracle, initial_inputs


if __name__ == "__main__":
    avicenna = Avicenna(
        grammar=grammar,
        initial_inputs=initial_inputs,
        oracle=oracle,
        max_iterations=10,
    )

    diagnoses = avicenna.explain()

    for diagnosis in diagnoses:
        print(ISLaUnparser(diagnosis[0]).unparse())
