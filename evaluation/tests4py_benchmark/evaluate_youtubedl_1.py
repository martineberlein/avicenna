import string
from typing import Union, Callable, Dict, Any

from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna_formalizations.tests4py import (
    setup_tests4py_project,
    DEFAULT_WORK_DIR,
    construct_oracle,
    run_oracle_check,
    get_tests4py_feature_learner
)


PROJECT_NAME: str = "youtubedl"
BUG_ID: int = 1
WORK_DIR = DEFAULT_WORK_DIR
setup_tests4py_project(PROJECT_NAME, BUG_ID, WORK_DIR)

oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
    PROJECT_NAME, BUG_ID, WORK_DIR
)


grammar = {
    "<start>": ["<match_str>"],
    "<match_str>": ["-q <stmt_list>\n-d {<dict_list>}"],
    "<stmt_list>": ["<stmt> & <stmt_list>", "<stmt>"],
    "<stmt>": ["<bool_stmt>", "<comp_stmt>"],
    "<bool_stmt>": ["<unary_op><name>"],
    "<unary_op>": ["!", ""],
    "<comp_stmt>": ["<name> <comp_op><optional> <int>"],
    "<optional>": ["?", ""],
    "<comp_op>": ["<", ">", "<=", ">=", "=", "!="],
    "<dict_list>": ["<kv>, <dict_list>", "<kv>", ""],
    "<kv>": ["<par><name><par>: <value>"],
    "<par>": ["'"],
    "<value>": ["<bool>", "<int>", "'<string>'", "''"],
    "<bool>": ["True", "False", "None"],
    "<name>": [
        "is_live",
        "like_count",
        "description",
        "title",
        "dislike_count",
        "test",
        "other",
    ],
    "<digit>": [str(i) for i in range(10)],
    "<int>": ["<int><digit>", "<digit>"],
    "<string>": ["<string><char>", "<char>"],
    "<char>": [str(char) for char in string.ascii_letters],
}


failing_list = [
    "-q !is_live\n-d {'is_live': False}",
    "-q !test\n-d {'test': False}",
    "-q !like_count & dislike_count <? 52 & description\n-d {'like_count': False, 'dislike_count': 10, 'description': ''}",
    "-q like_count > 134 & dislike_count <? 50 & !description\n-d {'like_count': 190, 'dislike_count': 23, 'description': False}",
    "-q like_count > 198 & !description\n-d {'like_count': 190, 'dislike_count': 4, 'description': False}",
    "-q !other & !description\n-d {'other': False, 'dislike_count': 1, 'description': False}",
    "-q !description\n-d {'other': False, 'dislike_count': 99999, 'description': False}",
    "-q !title\n-d {'title': False, 'description': False}",
    "-q description >? 914 & !title\n-d {'title': False}",
    "-q !is_live & description\n-d {'is_live': False, 'description': True}",
]


passing_list = [
    "-q test >? 0\n-d {}",
    "-q is_live\n-d {'is_live': None}",
    "-q !is_live\n-d {'is_live': None}",
    "-q !title\n-d {'title': ''}",
    "-q like_count > 100 & dislike_count <? 50 & description\n-d {'like_count': 190, 'dislike_count': 10}",
    "-q like_count > 100 & dislike_count <? 50 & description\n-d {'like_count': 190, 'dislike_count': 10, 'description': True}",
    "-q dislike_count >? 50 & description\n-d {'like_count': 190, 'dislike_count': 10, 'description': True}",
    "-q like_count > 100\n-d {'like_count': 190, 'title': False}",
    "-q !title\n-d {'title': 'abc'}",
    "-q is_live\n-d {}",
]

def eval_config() -> Dict[str, Any]:
    return {
        "grammar": grammar,
        "oracle": oracle,
        "initial_inputs": failing_list + passing_list,
        "feature_learner": get_tests4py_feature_learner(grammar),
    }


if __name__ == "__main__":
    run_checks = False
    if run_checks:
        run_oracle_check(oracle, failing_list, OracleResult.BUG)
        run_oracle_check(oracle, passing_list, OracleResult.NO_BUG)

    param = eval_config()
    avicenna = Avicenna(
        **param
    )

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())
