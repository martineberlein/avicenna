from typing import Dict, Any

from avicenna import Avicenna
from isla.language import ISLaUnparser

from avicenna.evaluation_setup import EvaluationSubject
from avicenna_formalizations.middle import grammar, oracle, initial_inputs


class MiddleSubject(EvaluationSubject):
    name = "Middle"

    def get_evaluation_config(self):
        param = self.default_param().copy()
        param.update(
            {
                "grammar": self.grammar,
                "oracle": self.oracle,
                "initial_inputs": self.initial_inputs,
                "top_n_relevant_features": 3,
            }
        )
        return param

    @classmethod
    def build(cls):
        return cls(grammar, oracle, initial_inputs)


if __name__ == "__main__":
    middle_subject = MiddleSubject.build()
    param = middle_subject.get_evaluation_config()

    avicenna = Avicenna(**param)

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    equivalent_representations = avicenna.get_equivalent_best_formulas()

    if equivalent_representations:
        for diagnosis in equivalent_representations:
            print(ISLaUnparser(diagnosis[0]).unparse())
