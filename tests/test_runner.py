import unittest

from debugging_benchmark.tests4py_benchmark.project import Calculator1Tests4PyProject, Fastapi2Tests4PyProject, Pysnooper2Tests4PyProject
from debugging_framework.docker.manager import DockerManager

from avicenna.data import Input
from avicenna.runner.execution_handler import BatchExecutionHandler


class TestExecutionRunner(unittest.TestCase):
  
    @unittest.skip("Skip for now")
    def test_batch_runner(self):
        subject = Pysnooper2Tests4PyProject()
        d_manager = DockerManager(subject.project)
        d_manager.build()
        d_manager.build_container(number_of_containers=5)

        def docker_oracle(input_list: list[Input]):
            inps = [str(inp) for inp in input_list]
            oracle_result = d_manager.run_inputs(inps)
            return oracle_result

        str_inputs = subject.failing_inputs + subject.passing_inputs
        inputs = set([Input.from_str(grammar=subject.grammar, input_string=inp) for inp in str_inputs])

        batch_runner = BatchExecutionHandler(oracle=docker_oracle)
        results = batch_runner.label(inputs)
        d_manager.cleanup()

        for inp in results:
            print(str(inp), inp.oracle)

if __name__ == '__main__':
    unittest.main()
