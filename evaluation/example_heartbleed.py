from typing import Dict, Any

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna_formalizations.heartbeat import grammar, oracle, initial_inputs
from avicenna.feature_extractor import DecisionTreeRelevanceLearner


def eval_config() -> Dict[str, Any]:
    return {
        "grammar": grammar,
        "oracle": oracle,
        "initial_inputs": initial_inputs,
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
