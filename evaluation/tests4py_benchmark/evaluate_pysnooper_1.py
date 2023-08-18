from typing import Union, Callable, Dict, Any

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna_formalizations.tests4py import (
    setup_tests4py_project,
    DEFAULT_WORK_DIR,
    construct_oracle,
    get_tests4py_feature_learner,
)

from avicenna_formalizations.tests4py import grammar_pysnooper as grammar

PROJECT_NAME: str = "pysnooper"
BUG_ID: int = 2
WORK_DIR = DEFAULT_WORK_DIR
setup_tests4py_project(PROJECT_NAME, BUG_ID, WORK_DIR)

oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
    PROJECT_NAME, BUG_ID, WORK_DIR
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


def eval_config() -> Dict[str, Any]:
    return {
        "grammar": grammar,
        "oracle": oracle,
        "initial_inputs": failing_list + passing_list,
        "feature_learner": get_tests4py_feature_learner(grammar),
    }


if __name__ == "__main__":
    param = eval_config()
    avicenna = Avicenna(**param)

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())
