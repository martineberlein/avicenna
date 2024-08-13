from avicenna import Avicenna

import evaluation.resources.seed as seed
from evaluation.resources.output import print_diagnoses

from debugging_benchmark.tests4py_benchmark.repository import (
    PysnooperBenchmarkRepository,
)

if __name__ == "__main__":
    programs = PysnooperBenchmarkRepository().build()
    for program in programs:
        param = program.to_dict()

        avicenna = Avicenna(**param, enable_logging=True)
        diagnoses = avicenna.explain()
        print_diagnoses(diagnoses)
