from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_benchmark.middle.middle import MiddleBenchmarkRepository
from debugging_framework.benchmark.program import BenchmarkProgram


def get_calculator_subject() -> BenchmarkProgram:
    """
    Get the calculator subject.
    :return: The calculator subject.
    """
    return CalculatorBenchmarkRepository().build()[0]


def get_middle_subject() -> BenchmarkProgram:
    """
    Get the middle subject.
    :return: The middle subject.
    """
    return MiddleBenchmarkRepository().build()[0]


def get_heartbleed_subject() -> BenchmarkProgram:
    """
    Get the heartbleed subject.
    :return: The heartbleed subject.
    """
    import importlib.util
    import os
    from pathlib import Path

    # Define the path to your module
    module_name = 'heartbeat'
    module_path = Path(__file__).parent.parent.parent / "evaluation" / "resources" / 'heartbeat.py'
    print(module_path)

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return BenchmarkProgram(
        name="Heartbleed",
        grammar=module.grammar,
        oracle=module.oracle,
        failing_inputs=module.failing_inputs,
        passing_inputs=module.passing_inputs,
    )


if __name__ == "__main__":
    prog = get_heartbleed_subject()
    print(prog.to_dict())

