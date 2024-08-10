from avicenna import Avicenna
from isla.language import ISLaUnparser
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository

if __name__ == "__main__":
    subjects = MiddleBenchmarkRepository().build()
    middle_subject = subjects[0]
    param = middle_subject.to_dict()

    avicenna = Avicenna(**param, max_iterations=10)
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
