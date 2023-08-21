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
    grammar_cookiecutter as grammar,
    get_tests4py_feature_learner,
)

PROJECT_NAME: str = "cookiecutter"
BUG_ID: int = 4
WORK_DIR = DEFAULT_WORK_DIR
setup_tests4py_project(PROJECT_NAME, BUG_ID, WORK_DIR)

oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
    PROJECT_NAME, BUG_ID, WORK_DIR
)


failing_list = [
    '{"full_name":["Marius Smytzek","Martin Eberlein"],"email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1',
    '{"full_name":"Marius Smytzek","email":["mariussmytzek@cispa.de","martineberlein@huberlin.de"],"github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npost:exit,0',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":["t4p","p4t"],"project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}',
    '{"full_name":["Marius Smytzek","Martin Eberlein","Michael Mera"],"email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npost:echo,post1',
    '{"full_name":["Marius Smytzek","Martin Eberlein","Michael Mera"],"email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":["t4p","p4t"],"project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npost:exit,0',
    '{"full_name":"Marius Smytzek","email":["mariussmytzek@cispa.de","martineberlein@huberlin.de"],"github_username":"smythi93","project_name":"Test4Py Project","repo_name":["t4p","p4t"],"project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}',
    '{"full_name":["Marius Smytzek","Martin Eberlein","Michael Mera"],"email":["mariussmytzek@cispa.de","martineberlein@huberlin.de"],"github_username":"smythi93","project_name":"Test4Py Project","repo_name":["t4p","p4t"],"project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":["2022","2023"],"version":"0.1"}\npost:echo,post1',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":["2022","2023","2024","2025"],"version":"0.1"}\npre:echo,pre1',
    '{"full_name":["Marius Smytzek","Martin Eberlein","Michael Mera"],"email":["mariussmytzek@cispa.de","martineberlein@huberlin.de"],"project_name":["tests4py","py4tests"],"repo_name":["t4p","p4t"],"project_short_description":["description","No description"],"release_date":["2022-06-23","2023-01-22"],"year":["2022","2023","2024","2025"],"version":["0.3","1.8.2"]}\npre:echo,This is a more complex example of a pre hook_ Will this work',
]


passing_list = [
    '{"full_name":"Martin Eberlein","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1',
    '{"full_name":"Marius Smytzek","email":"martineberlein@huberlin.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npost:exit,0',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"p4t","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\n',
    '{"full_name":"Michael Mera","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npost:echo,post1',
    '{"full_name":"Michael Mera","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"p4t","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npost:exit,0',
    '{"full_name":"Marius Smytzek","email":"martineberlein@huberlin.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"p4t","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\n',
    '{"full_name":"Michael Mera","email":"martineberlein@huberlin.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"p4t","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\n',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2023","version":"0.1"}\npost:echo,post1',
    '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2025","version":"0.1"}\npre:echo,pre1',
    '{"full_name":"Michael Mera","email":"martineberlein@huberlin.de","github_username":"smythi93","project_name":"py4tests","repo_name":"p4t","project_short_description":"No description","release_date":"2023-01-22","year":"2025","version":"1.8.2"}\npre:echo,This is a more complex example of a pre hook_ Will this work',
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
