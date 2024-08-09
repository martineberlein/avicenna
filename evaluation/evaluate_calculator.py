from isla.language import ISLaUnparser

from avicenna.avicenna_new import Avicenna
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from avicenna.learning.heuristic import HeuristicTreePatternLearner


if __name__ == "__main__":
    default_param = {
        "max_iterations": 10,
    }

    calculator_subject = CalculatorBenchmarkRepository().build()
    param = calculator_subject[0].to_dict()
    param.update(default_param)

    avicenna = Avicenna(**param)
    diagnoses = avicenna.explain()

    diagnosis = diagnoses.pop(0)
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis.formula).unparse())
    print(f"Precision: {diagnosis.precision()} Recall: {diagnosis.recall()} Length: {len(diagnosis.formula)}")

    print("\nEquivalent Representations:")
    equivalent_representations = diagnoses

    if equivalent_representations:
        for diagnosis in equivalent_representations:
            print(ISLaUnparser(diagnosis.formula).unparse())
            print(f"Precision: {diagnosis.precision()} Recall: {diagnosis.recall()} Length: {len(diagnosis.formula)}")
