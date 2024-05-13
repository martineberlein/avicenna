import string
from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.input import OracleResult
from avicenna.pattern_learner import AvicennaPatternLearner


if __name__ == "__main__":
    from debugging_benchmark.tests4py_benchmark.repository import (
        PysnooperBenchmarkRepository,
    )

    programs = PysnooperBenchmarkRepository().build()
    for program in programs:

        param = program.to_dict()
        avicenna = Avicenna(
            **param,
            # pattern_learner=AvicennaPatternLearner
        )

        diagnosis = avicenna.explain()
        print(f"Final Diagnosis for {program}:")
        print(ISLaUnparser(diagnosis[0]).unparse())

        equivalent_representations = avicenna.get_equivalent_best_formulas()

        if equivalent_representations:
            print("\nEquivalent Representations:")
            for diagnosis in equivalent_representations:
                print(ISLaUnparser(diagnosis[0]).unparse(), end="\n\n")

        print(f"All Learned Formulas (that meet min criteria) for {program}:")
        cand = avicenna.get_learned_formulas()
        for can in cand:
            print(
                f"Avicenna calculated a precision of {can[1] * 100:.2f}% and a recall of {can[2] * 100:.2f}%"
            )
            print(ISLaUnparser(can[0]).unparse(), end="\n\n")
