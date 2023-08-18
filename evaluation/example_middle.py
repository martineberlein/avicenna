from typing import Dict, Any

from avicenna import Avicenna
from isla.language import ISLaUnparser

from avicenna_formalizations.middle import grammar, oracle, initial_inputs


def eval_config() -> Dict[str, Any]:
    return {
        "grammar": grammar,
        "oracle": oracle,
        "initial_inputs": initial_inputs,
        "max_excluded_features": 4,
        "max_iterations": 20,
    }


if __name__ == "__main__":

    param = eval_config()
    avicenna = Avicenna(
        **param
    )

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())
