from typing import List

from isla.language import ISLaUnparser

from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.student_assignments import (
    MiddleAssignmentBenchmarkRepository,
    GCDStudentAssignmentBenchmarkRepository,
    NPrStudentAssignmentBenchmarkRepository
)
from debugging_framework.benchmark import BenchmarkRepository, BenchmarkProgram

from avicenna import Avicenna
import avicenna.pattern_learner as pattern_learner


patterns = [
"""exists <?NONTERMINAL> elem_1 in start:
  exists <?NONTERMINAL> elem_2 in start:
    (= (str.to.int elem_1) (str.to.int elem_2))""",
    """
    exists <?NONTERMINAL> elem in start:
    (>= (str.to.int elem) (str.to.int <?STRING>))
    """
]


def main():
    repos: List[BenchmarkRepository] = [MiddleAssignmentBenchmarkRepository()]

    subjects: List[BenchmarkProgram] = []
    for repo in repos:
        subjects_ = repo.build()
        subjects += subjects_

    print(f"Number of subjects: {len(subjects)}")

    for subject in subjects:
        param = subject.to_dict()
        param.update(
            {
                "top_n_relevant_features": 3,
                "max_iterations": 10,
                "log": True,
                #"pattern_learner": pattern_learner.AviIslearn,
                #"patterns": patterns,
            }
        )
        try:
            avicenna = Avicenna(**param)
            diagnosis = avicenna.explain()
            if diagnosis:
                print(f"Final Diagnosis for {subject}")
                print(ISLaUnparser(diagnosis[0]).unparse())
                print(
                    f"Avicenna calculated a precision: {diagnosis[1] * 100:.2f}% and recall {diagnosis[2] * 100:.2f}%"
                )
            else:
                print(f"No diagnosis has been learned for {subject}!")
        except Exception as e:
            print(f"Could not learn diagnosis for {subject}")


if __name__ == "__main__":
    main()
