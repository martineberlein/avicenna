from typing import Dict, Any

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.feature_extractor import DecisionTreeRelevanceLearner
from avicenna_formalizations.calculator import grammar, oracle, initial_inputs


def eval_config() -> Dict[str, Any]:
    return {
        "grammar": grammar,
        "oracle": oracle,
        "initial_inputs": initial_inputs,
        "feature_learner": DecisionTreeRelevanceLearner(grammar)
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
