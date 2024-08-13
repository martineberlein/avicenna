from avicenna import Avicenna

from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from evaluation.resources.output import print_diagnoses

if __name__ == "__main__":
    subjects = MiddleBenchmarkRepository().build()
    middle_subject = subjects[0]
    param = middle_subject.to_dict()

    avicenna = Avicenna(**param, enable_logging=True)
    diagnoses = avicenna.explain()
    print_diagnoses(diagnoses)
