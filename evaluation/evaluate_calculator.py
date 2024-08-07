from isla.language import ISLaUnparser

from avicenna import Avicenna
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from avicenna.learning.heuristic import HeuristicTreePatternLearner


if __name__ == "__main__":
    default_param = {
        "log": True,
        "max_iterations": 4,
    }

    calculator_subject = CalculatorBenchmarkRepository().build()
    param = calculator_subject[0].to_dict()
    param.update(default_param)

    avicenna = Avicenna(**param)

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis.formula).unparse())
    print(f"Precision: {diagnosis.precision()} Recall: {diagnosis.recall()} Length: {len(diagnosis.formula)}")

    print("\nEquivalent Representations:")
    equivalent_representations = avicenna.get_equivalent_best_formulas()

    if equivalent_representations:
        for diagnosis in equivalent_representations:
            print(ISLaUnparser(diagnosis.formula).unparse())
            print(f"Precision: {diagnosis.precision()} Recall: {diagnosis.recall()} Length: {len(diagnosis.formula)}")
