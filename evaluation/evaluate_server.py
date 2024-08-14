from typing import List
import os
import datetime

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.learning.table import Candidate
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.tests4py_benchmark.repository import (
    CalculatorBenchmarkRepository as CalculatorTests4PyBenchmarkRepository,
    MiddleBenchmarkRepository,
    MarkUpBenchmarkRepository,
    ExpressionBenchmarkRepository,
    PysnooperBenchmarkRepository,
    CookieCutterBenchmarkRepository,
)
from debugging_framework.benchmark.repository import BenchmarkRepository
from debugging_framework.benchmark.program import BenchmarkProgram


EvalDict = {
    "middle_1": {
        "top_n_relevant_features": 3,
    },
    "middle_2": {
        "top_n_relevant_features": 3,
    },
    "cookiecutter_2": {"min_recall": 0.7},
    "cookiecutter_3": {"min_recall": 0.7},
    "cookiecutter_4": {"min_recall": 0.7},
}


if __name__ == "__main__":
    # Get the current timestamp and format it
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the output directory with the timestamp
    output_dir = os.path.join(script_dir, f"diagnosis_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    repos: List[BenchmarkRepository] = [
        CalculatorBenchmarkRepository(),
        CalculatorTests4PyBenchmarkRepository(),
        MiddleBenchmarkRepository(),
        MarkUpBenchmarkRepository(),
        ExpressionBenchmarkRepository(),
        PysnooperBenchmarkRepository(),
        CookieCutterBenchmarkRepository(),
    ]

    programs: List[BenchmarkProgram] = []
    for repo in repos:
        for prog in repo.build():
            programs.append(prog)

    for program in programs:
        print(f"Starting Avicenna for {program}")

        param = program.to_dict()
        # load additional evaluation parameter
        default_param = EvalDict.get(program.name, {})  # Use {} as the default value
        param.update(default_param)

        avicenna = Avicenna(
            **param,
        )

        program_name = str(program).replace(" ", "_")
        output_file = os.path.join(output_dir, f"{program_name}_diagnosis.txt")

        with open(output_file, "w") as f:
            try:
                diagnoses: List[Candidate] = avicenna.explain()
                if diagnoses:
                    for diagnosis in diagnoses:
                        f.write(
                            f"Avicenna calculated a precision of {diagnosis.precision() * 100:.2f}%, "
                            f"recall of {diagnosis.recall() * 100:.2f}% "
                            f"and specificity of {diagnosis.precision() * 100:.2f}%\n"
                        )
                        f.write(ISLaUnparser(diagnosis.formula).unparse() + "\n\n")
                else:
                    f.write(f"No Diagnosis for {program}!\n\n")

            except Exception as e:
                f.write(f"An error occurred while diagnosing {program}: {e}\n\n")
                print(f"An error occurred while diagnosing {program}: {e}")

        print(f"Diagnosis saved to {output_file}")
