from typing import List, Optional
from isla.language import ISLaUnparser

from avicenna.learning.table import Candidate


def print_diagnoses(diagnoses: Optional[List[Candidate]]):
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
