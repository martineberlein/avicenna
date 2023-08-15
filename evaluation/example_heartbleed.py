from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna_formalizations.heartbeat import grammar, oracle, initial_inputs


if __name__ == "__main__":
    avicenna = Avicenna(
        grammar=grammar,
        initial_inputs=initial_inputs,
        oracle=oracle,
        max_iterations=5,
        log=True
    )

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())

