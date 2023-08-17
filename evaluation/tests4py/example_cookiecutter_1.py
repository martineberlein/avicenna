import string
from typing import Union, Callable

from isla.language import ISLaUnparser
from fuzzingbook.Grammars import srange

from avicenna import Avicenna
from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna_formalizations.tests4py import (
    setup_tests4py_project,
    DEFAULT_WORK_DIR,
    construct_oracle,
    run_oracle_check,
)


grammar = {
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


initial_inputs = failing_list + passing_list


if __name__ == "__main__":
    project_name: str = "cookiecutter"
    bug_id: int = 2
    work_dir = DEFAULT_WORK_DIR
    setup_tests4py_project(project_name, bug_id, work_dir)

    oracle: Callable[[Union[str, Input]], OracleResult] = construct_oracle(
        project_name, bug_id, work_dir
    )

    from tests4py import framework

    failing = '{"full_name":"Marius Smytzek","email":"mariussmtzek@cispa.de","github_username":"smythi93","project_name":"Test4Py Project","repo_name":"t4p","project_short_description":"The t4p project","release_date":"2022-12-25","year":"2022","version":"0.1"}\npre:echo,pre1\npre:echo,pre2'

    report = framework.systemtest.tests4py_test(
        work_dir=work_dir / "cookiecutter_2", path_or_str=failing, diversity=False
    )
    print(report)
    print(report.raised)

    run_oracle_check(oracle, failing_list, OracleResult.BUG)
    # run_oracle_check(oracle, passing_list, OracleResult.NO_BUG)
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
