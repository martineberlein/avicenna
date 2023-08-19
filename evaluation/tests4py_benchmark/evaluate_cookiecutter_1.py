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
    get_tests4py_feature_learner,
    grammar_cookiecutter as grammar,
)

PROJECT_NAME: str = "cookiecutter"
BUG_ID: int = 2
WORK_DIR = DEFAULT_WORK_DIR
setup_tests4py_project(PROJECT_NAME, BUG_ID, WORK_DIR)

oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
    PROJECT_NAME, BUG_ID, WORK_DIR
)


failing_list = [
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npre:echo,pre2',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npost:echo,post1\npost:echo,post2',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npost:echo,post2\npre:echo,pre2',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npost:echo,post1\npost:echo,post2\npre:echo,pre1',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npost:echo,post1\npre:echo,pre2\npost:echo,post2',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npre:echo,pre2\npre:echo,pre3',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npost:echo,post1\npost:echo,post2\npost:echo,post3',
    '{"full_name":["Marius Smytzek","Martin Eberlein"],"email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npost:echo,post1\npost:echo,post2',
    '{"full_name":["Marius Smytzek","Martin Eberlein"],"email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npre:echo,pre2',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,This is a more complex example of a pre hook_ Will this work\npre:echo,pre2',
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
    avicenna = Avicenna(**param)

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    print("\nEquivalent Representations:")
    for diagnosis in avicenna.get_equivalent_best_formulas():
        print(ISLaUnparser(diagnosis[0]).unparse())
