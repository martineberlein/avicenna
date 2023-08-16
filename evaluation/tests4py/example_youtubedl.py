from typing import Union, Callable

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna_formalizations.tests4py import (
    setup_tests4py_project,
    DEFAULT_WORK_DIR,
    construct_oracle,
)


failing_list = [
    "-otest.log\n-cint=str\n",
    "-cint=str\n",
    "-d1\n-cint=repr\n-T\n",
    "-o\n-d1\n-cbool=str\n",
    "-otest.log\n-cint=repr,bool=str\n-O\n",
    "-d1\n-wx\n-cfloat=str\n",
    "-wy\n-cstr=str\n",
    "-otest.log\n-wx\n-cstr=int\n",
    "-ptest\n-cbool=int\n",
    "-ptest\n-wx\n-cint=str\n",
]

passing_list = [
    "-otest.log\n",
    "",
    "-d1\n-T\n",
    "-o\n-d1\n",
    "-otest.log\n-O\n",
    "-d1\n-wx\n",
    "-wy\n",
    "-otest.log\n-wx\n",
    "-ptest\n",
    "-ptest\n-wx\n",
]

initial_inputs = failing_list + passing_list


if __name__ == "__main__":
    from avicenna_formalizations.tests4py import grammar_pysnooper as grammar

    project_name: str = "youtubedl"
    bug_id: int = 1
    work_dir = DEFAULT_WORK_DIR
    setup_tests4py_project(project_name, bug_id, work_dir)

    oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
        project_name, bug_id, work_dir
    )
    # for inp in failing_list:
    #     print(oracle(inp))

    from tests4py import framework
    report = framework.systemtest.tests4py_test(
        work_dir=work_dir / f"youtubedl_1", path_or_str="-q !is_live\n-d {'is_live': False}", diversity=False
    )
    print(report.failing)
    print(report.results)


    exit()
    avicenna = Avicenna(
        grammar=grammar,
        initial_inputs=initial_inputs,
        oracle=oracle,
        max_iterations=3,
        log=True,
    )

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())
    