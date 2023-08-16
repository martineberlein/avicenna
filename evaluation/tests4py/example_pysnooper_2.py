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
    "-otest1.log\n",
    "-otest2.log\n-d1\n",
    "-otest3.log\n-vx\n",
    "-otest4.log\n-vx,y\n",
    "-otest5.log\n-vw\n-d2\n",
    "-otest6.log\n-ptest\n",
    "-otest7.log\n-d1\n-ptest\n",
    "-otest8.log\n-vx\n-ptest\n",
    "-otest9.log\n-vw,x,y,z\n-d1\n",
    "-otest10.log\n-vx,z\n-d1\n-ptest\n",
]


passing_list = [
    "-o\n-d1\n",
    "-vx\n",
    "-o\n-vx,y\n",
    "-vw\n-d2\n",
    "-o\n-ptest\n",
    "-d1\n-ptest\n",
    "-o\n-vx\n-ptest\n",
    "-vw,x,y,z\n-d1\n",
    "-o\n-vx,z\n-d1\n-ptest\n",
]


initial_inputs = failing_list + passing_list


if __name__ == "__main__":
    from avicenna_formalizations.tests4py import grammar_pysnooper as grammar

    project_name: str = "pysnooper"
    bug_id: int = 3
    work_dir = DEFAULT_WORK_DIR
    setup_tests4py_project(project_name, bug_id, work_dir)

    oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
        project_name, bug_id, work_dir
    )

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
