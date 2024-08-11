from avicenna import Avicenna

from evaluation.resources.heartbeat import grammar, oracle, initial_inputs
from evaluation.resources.output import print_diagnoses

if __name__ == "__main__":
    param = {
        "grammar": grammar,
        "initial_inputs": initial_inputs,
        "oracle": oracle,
    }

    avicenna = Avicenna(**param, enable_logging=True)
    diagnoses = avicenna.explain()
    print_diagnoses(diagnoses)
