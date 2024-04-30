from isla.language import ISLaUnparser

from avicenna import Avicenna
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository


if __name__ == "__main__":
    default_param = {
        "log": True,
        "max_iterations": 20,
    }

    calculator_subject = CalculatorBenchmarkRepository().build()
    param = calculator_subject[0].to_dict()
    param.update(default_param)

    avicenna = Avicenna(**param)

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    equivalent_representations = avicenna.get_equivalent_best_formulas()

    if equivalent_representations:
        for diagnosis in equivalent_representations:
            print(ISLaUnparser(diagnosis[0]).unparse())
