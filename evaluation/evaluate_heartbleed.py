from typing import Dict, Any

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna_formalizations.heartbeat import grammar, oracle, initial_inputs
from avicenna.evaluation_setup import EvaluationSubject


def eval_config() -> Dict[str, Any]:
    return {
        "grammar": grammar,
        "oracle": oracle,
        "initial_inputs": initial_inputs,
    }


class HeartbleedSubject(EvaluationSubject):
    name = "Heartbleed"

    @classmethod
    def build(cls):
        return cls(grammar, oracle, initial_inputs)


if __name__ == "__main__":
    heartbleed_subject = HeartbleedSubject.build()
    param = heartbleed_subject.get_evaluation_config()

    avicenna = Avicenna(**param)

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())
