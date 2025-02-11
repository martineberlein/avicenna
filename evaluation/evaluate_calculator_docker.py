from avicenna import Avicenna
from avicenna.data import Input
from avicenna.runner.execution_handler import BatchExecutionHandler

from debugging_benchmark.tests4py_benchmark.project import Calculator1Tests4PyProject
from debugging_benchmark.calculator.calculator import CalculatorBenchmarkRepository
from debugging_framework.docker.manager import DockerManager

from evaluation.resources.output import print_diagnoses


if __name__ == "__main__":
    subject = Calculator1Tests4PyProject()
    d_manager = DockerManager(subject.project)
    d_manager.build()
    d_manager.build_container(number_of_containers=7)

    def docker_oracle(input_list: list[Input]):
        inps = [str(inp) for inp in input_list]
        oracle_result = d_manager.run_inputs(inps)
        return oracle_result


    batch_runner = BatchExecutionHandler(oracle=docker_oracle)

    default_param = {
        "max_iterations": 10,
        "oracle": docker_oracle,
    }

    calculator_subject = CalculatorBenchmarkRepository().build()
    param = calculator_subject[0].to_dict()
    param.update(default_param)

    avicenna = Avicenna(**param, enable_logging=True, runner=batch_runner)
    diagnoses = avicenna.explain()
    print_diagnoses(diagnoses)
