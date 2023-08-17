from typing import Union, Callable

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna_formalizations.tests4py import (
    setup_tests4py_project,
    DEFAULT_WORK_DIR,
    construct_oracle,
    run_oracle_check,
    run_parsing_checks,
    get_tests4py_feature_learner,
)


grammar = {
    "<start>": ["<url>"],
    "<url>": ["<scheme>://<authority><path_to_file><query>"],
    "<scheme>": ["http", "https", "ftp", "ftps"],
    "<authority>": ["<host>", "<host>:<port>"],
    "<host>": ["<host_name>.<tld>"],
    "<host_name>": ["media.w3", "foo", "bar"],
    "<tld>": ["de", "com", "org"],
    "<port>": ["80", "8080", "<nat>"],
    "<nat>": ["<digit>", "<digit><digit>"],
    "<digit>": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<path_to_file>": ["<path>", "<path_with_file>"],
    "<path_with_file>": ["<path>/<file>"],
    "<path>": ["", "/", "<path_><path>"],
    "<path_>": ["/<id>"],
    "<id>": [
        "abc",
        "def",
        "x<digit><digit>",
        "foo",
        "bar",
        "baz",
        "2010",
        "05",
        "sintel",
        "x",
        "y",
    ],
    "<file>": ["<file_name>.<file_extension>"],
    "<file_name>": ["trailer", "foo", "bar"],
    "<file_extension>": ["mp4", "pdf", "dot"],
    "<query>": ["", "?<params>", "#<params>"],
    "<params>": ["<param>", "<param>&<params>"],
    "<param>": ["<id>=<id>", "<id>=<nat>"],
}

failing_list = [
    "http://media.w3.org/2010/05/sintel/trailer.mp4",
]


passing_list = [
    "http://foo.de/bar/baz/",
    "http://foo.de/bar/baz#x=y",
    "http://foo.de/bar/baz?x=y",
    "http://foo.de/bar/baz",
    "http://foo.de/",
    "http://media.w3.org/trailer.mp4",
]

initial_inputs = failing_list + passing_list


if __name__ == "__main__":
    project_name: str = "youtubedl"
    bug_id: int = 43
    work_dir = DEFAULT_WORK_DIR
    setup_tests4py_project(project_name, bug_id, work_dir)

    oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
        project_name, bug_id, work_dir
    )

    run_parsing_checks(grammar, input_list=initial_inputs)
    run_oracle_check(oracle, failing_list, OracleResult.BUG)
    run_oracle_check(oracle, passing_list, OracleResult.NO_BUG)

    avicenna = Avicenna(
        grammar=grammar,
        initial_inputs=initial_inputs,
        oracle=oracle,
        max_iterations=20,
        log=True,
        feature_learner=get_tests4py_feature_learner(grammar)
    )

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())
