from pathlib import Path
import string
import os

from isla.language import ISLaUnparser


from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input

from tests4py.grammars.fuzzer import GrammarFuzzer, Grammar, srange, is_valid_grammar
from tests4py import framework

grammar: Grammar = {
    "<start>": ["<options>"],
    "<options>": ["<output><variables><depth><prefix><watch><custom_repr><overwrite><thread_info>"],

    "<output>": ["-o\n", "-o<path>\n", ""],
    "<variables>": ["-v<variable_list>\n", ""],
    "<depth>": ["-d<int>\n", ""],
    "<prefix>": ["-p<str>\n", ""],
    "<watch>": ["-w<variable_list>\n", ""],
    "<custom_repr>": ["-c<predicate_list>\n", ""],
    "<overwrite>": ["-O\n", ""],
    "<thread_info>": ["-T\n", ""],
    "<path>": ["<location>", "<location>.<str>"],
    "<location>": ["<str>", os.path.join("<path>", "<str>")],
    "<variable_list>": ["<variable>", "<variable_list>,<variable>"],
    "<variable>": ["<name>", "<variable>.<name>"],
    "<name>": ["<letter><chars>"],
    "<chars>": ["", "<chars><char>"],
    "<letter>": srange(string.ascii_letters),
    "<digit>": srange(string.digits),
    "<char>": ["<letter>", "<digit>", "_"],
    "<int>": ["<nonzero><digits>", "0"],
    "<digits>": ["", "<digits><digit>"],
    "<nonzero>": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<str>": ["<char><chars>"],
    "<predicate_list>": ["<predicate>", "<predicate_list>,<predicate>"],
    "<predicate>": ["<p_function>=<t_function>"],
    "<p_function>": ["int", "str", "float", "bool"],
    "<t_function>": ["repr", "str", "int"],
}

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
    "-ptest\n-wx\n-cint=str\n"
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
    "-ptest\n-wx\n"
]

initial_inputs = failing_list + passing_list

def oracle(inp: str | Input):
    path = Path("/Users/martineberlein/github/avicenna/notebooks/tests4py/tmp/pysnooper_2").absolute()
    report = framework.systemtest.tests4py_test(work_dir=path, path_or_str=str(inp), diversity=False)
    print(report.failing, report.passing)
    print(path)
    if report.failing == 1:
        return OracleResult.BUG
    elif report.passing == 1:
        return OracleResult.NO_BUG
    return OracleResult.UNDEF



if __name__ == "__main__":
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

