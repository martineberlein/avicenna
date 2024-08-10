from isla.language import ISLaUnparser

from avicenna import Avicenna
from evaluation.resources.heartbeat import grammar, oracle, initial_inputs


if __name__ == "__main__":
    param = {
        "grammar": grammar,
        "initial_inputs": initial_inputs,
        "oracle": oracle,
    }

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
