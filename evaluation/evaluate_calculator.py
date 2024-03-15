from typing import Dict, Any

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.feature_extractor import DecisionTreeRelevanceLearner
from avicenna_formalizations.calculator import grammar, oracle, initial_inputs
from avicenna.evaluation_setup import EvaluationSubject
from avicenna.generator import ISLaSolverGenerator


class CalculatorSubject(EvaluationSubject):
    name = "Calculator"

    @classmethod
    def build(cls):
        return cls(grammar, oracle, initial_inputs)


if __name__ == "__main__":
    calculator_subject = CalculatorSubject.build()
    param = calculator_subject.get_evaluation_config()

    avicenna = Avicenna(**param)

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    equivalent_representations = avicenna.get_equivalent_best_formulas()

    if equivalent_representations:
        for diagnosis in equivalent_representations:
            print(ISLaUnparser(diagnosis[0]).unparse())
