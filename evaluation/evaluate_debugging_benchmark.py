from typing import List

from isla.language import ISLaUnparser

from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.student_assignments import MiddleAssignmentBenchmarkRepository
from debugging_framework.benchmark import BenchmarkRepository, BenchmarkProgram

from avicenna import Avicenna



def main():

    repos: List[BenchmarkRepository] = [
        CalculatorBenchmarkRepository()
    ]

    subjects: List[BenchmarkProgram] = []
    for repo in repos:
        subjects_ = repo.build()
        subjects += subjects_

    for subject in subjects:
        param = subject.to_dict()

        avicenna = Avicenna(**param)
        diagnosis = avicenna.explain()
        print(f"Final Diagnosis for {subject}")
        print(ISLaUnparser(diagnosis[0]).unparse())


if __name__ == "__main__":
    main()