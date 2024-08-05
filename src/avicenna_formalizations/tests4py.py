import os
import string
from pathlib import Path
from typing import Union, Callable, List

from fuzzingbook.Parser import EarleyParser
from fuzzingbook.Grammars import srange, Grammar
# from tests4py.grammars.fuzzer import GrammarFuzzer, Grammar, srange, is_valid_grammar
from tests4py import framework
from tests4py.projects import load_bug_info
import tests4py.constants

from avicenna.input import Input
from avicenna.oracle import OracleResult
from avicenna.features.feature_extractor import DecisionTreeRelevanceLearner


DEFAULT_WORK_DIR = Path("/tmp")


def setup_tests4py_project(
    project_name: str, bug_id: int, work_dir: Path = DEFAULT_WORK_DIR
):
    report = framework.default.tests4py_checkout(
        project_name=project_name, bug_id=bug_id, work_dir=work_dir
    )
    if report.raised:
        raise report.raised

    project_dir = work_dir / f"{project_name}_{bug_id}"
    assert project_dir.exists()

    report = framework.default.tests4py_compile(project_dir)
    if report.raised:
        raise report.raised
    project = load_bug_info(project_dir / tests4py.constants.INFO_FILE)
    assert project.compiled


def generic_tests4py_oracle(inp_: str | Input, project_dir: Path = DEFAULT_WORK_DIR):
    report = framework.systemtest.tests4py_test(
        work_dir=project_dir, path_or_str=str(inp_), diversity=False
    )
    if report.failing == 1:
        return OracleResult.BUG
    elif report.passing == 1:
        return OracleResult.NO_BUG
    return OracleResult.UNDEF


def construct_oracle(
    project_name: str, bug_id: int, work_dir: Path = DEFAULT_WORK_DIR
) -> Callable[[Union[str, Input]], OracleResult]:
    def oracle(inp: Union[str, Input]) -> OracleResult:
        project_dir = work_dir / f"{project_name}_{bug_id}"
        return generic_tests4py_oracle(inp, project_dir)

    return oracle


def run_parsing_checks(grammar: Grammar, input_list: List[str]):
    parser = EarleyParser(grammar)
    for inp in input_list:
        for tree in parser.parse(inp):
            pass


def run_oracle_check(
    oracle: Callable, input_list: List[str], expected_result: OracleResult
):
    for inp in input_list:
        assert oracle(inp) == expected_result

def get_tests4py_feature_learner(grammar: Grammar):
    return DecisionTreeRelevanceLearner(
        grammar
    )


grammar_pysnooper: Grammar = {
    "<start>": ["<options>"],
    "<options>": [
        "<output><variables><depth><prefix><watch><custom_repr><overwrite><thread_info>"
    ],
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

grammar_cookiecutter: Grammar = {
    "<start>": ["<config>\n<hooks>"],
    "<config>": ["{<pairs>}", "{}"],
    "<hooks>": ["", "<hook_list>"],
    "<hook_list>": ["<hook>", "<hook_list>\n<hook>"],
    "<hook>": ["<pre_hook>", "<post_hook>"],
    "<pre_hook>": ["pre:<hook_content>"],
    "<post_hook>": ["post:<hook_content>"],
    "<hook_content>": ["echo,<str_with_spaces>", "exit,<int>"],
    "<pairs>": ["<pair>", "<pairs>,<pair>"],
    "<pair>": [
        "<full_name>",
        "<email>",
        "<github_username>",
        "<project_name>",
        "<repo_name>",
        "<project_short_description>",
        "<release_date>",
        "<year>",
        "<version>",
    ],
    "<full_name>": [
        '"full_name":"<str_with_spaces>"',
        '"full_name":[<str_with_spaces_list>]',
    ],
    "<email>": ['"email":"<email_address>"', '"email":[<email_list>]'],
    "<github_username>": [
        '"github_username":"<str>"',
        '"github_username":[<str_list>]',
    ],
    "<project_name>": [
        '"project_name":"<str_with_spaces>"',
        '"project_name":[<str_with_spaces_list>]',
    ],
    "<repo_name>": ['"repo_name":"<str>"', '"repo_name":[<str_list>]'],
    "<project_short_description>": [
        '"project_short_description":"<str_with_spaces>"',
        '"project_short_description":[<str_with_spaces_list>]',
    ],
    "<release_date>": ['"release_date":"<date>"', '"release_date":[<date_list>]'],
    "<year>": ['"year":"<int>"', '"year":[<int_list>]'],
    "<version>": ['"version":"<v>"', '"version":[<version_list>]'],
    "<str_with_spaces_list>": [
        '"<str_with_spaces>"',
        '<str_with_spaces_list>,"<str_with_spaces>"',
    ],
    "<email_list>": ['"<email_address>"', '<email_list>,"<email_address>"'],
    "<str_list>": ['"<str>"', '<str_list>,"<str>"'],
    "<int_list>": ['"<int>"', '<int_list>,"<int>"'],
    "<date_list>": ['"<date>"', '<date_list>,"<date>"'],
    "<version_list>": ['"<v>"', '<version_list>,"<v>"'],
    "<chars>": ["", "<chars><char>"],
    "<char>": srange(string.ascii_letters + string.digits + "_"),
    "<chars_with_spaces>": ["", "<chars_with_spaces><char_with_spaces>"],
    "<char_with_spaces>": srange(string.ascii_letters + string.digits + "_ "),
    "<str>": ["<char><chars>"],
    "<str_with_spaces>": ["<char_with_spaces><chars_with_spaces>"],
    "<email_address>": ["<str>@<str>.<str>"],
    "<date>": ["<day>.<month>.<int>", "<int>-<month>-<day>"],
    "<month>": ["0<nonzero>", "<nonzero>", "10", "11", "12"],
    "<day>": [
        "0<nonzero>",
        "<nonzero>",
        "10",
        "1<nonzero>",
        "20",
        "2<nonzero>",
        "30",
        "31",
    ],
    "<v>": ["<digit><digits>", "<v>.<digit><digits>"],
    "<int>": ["<nonzero><digits>", "0"],
    "<digits>": ["", "<digits><digit>"],
    "<nonzero>": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<digit>": srange(string.digits),
}