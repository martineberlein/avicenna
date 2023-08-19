from typing import Union, Callable

from isla.language import ISLaUnparser
from tests4py.projects.fastapi import grammar_request

from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna_formalizations.tests4py import (
    setup_tests4py_project,
    DEFAULT_WORK_DIR,
    construct_oracle,
    run_oracle_check,
    run_parsing_checks,
)

failing_list = ["-o{'foo':'test'}\n-d\n"]

passing_list = ["-o{'foo':'test'}"]


if __name__ == "__main__":
    project_name: str = "fastapi"
    bug_id: int = 4
    work_dir = DEFAULT_WORK_DIR
    setup_tests4py_project(project_name, bug_id, work_dir)

    oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
        project_name, bug_id, work_dir
    )
    from tests4py import framework

    report = framework.systemtest.tests4py_test(
        work_dir=work_dir / "fastapi_1",
        path_or_str=str(passing_list[0]),
        diversity=False,
    )
    print(report.results)

    # run_parsing_checks(grammar, input_list=initial_inputs)
    run_oracle_check(oracle, failing_list, OracleResult.BUG)
    # run_oracle_check(oracle, passing_list, OracleResult.NO_BUG)
    #
    # avicenna = Avicenna(
    #     grammar=grammar,
    #     initial_inputs=initial_inputs,
    #     oracle=oracle,
    #     max_iterations=10,
    #     log=True,
    # )
    #
    # diagnosis = avicenna.explain()
    # print("Final Diagnosis:")
    # print(ISLaUnparser(diagnosis[0]).unparse())
    #
    # print("\nEquivalent Representations:")
    # for diagnosis in avicenna.get_equivalent_best_formulas():
    #     print(ISLaUnparser(diagnosis[0]).unparse())
