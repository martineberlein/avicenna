import queue
import string
import subprocess
import traceback
from abc import ABC
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple

from fuzzingbook.Grammars import Grammar, is_valid_grammar, srange
from isla.derivation_tree import DerivationTree

from avicenna_formalizations.t4p_common import (
    Environment,
    HARNESS_FILE,
    API,
    ExpectErrAPI,
    TestResult,
    GrammarVisitor,
)


class FastAPI1API(ExpectErrAPI):
    pass


class FastAPIDefaultAPI(API, GrammarVisitor):
    """ """

    def __init__(self, default_timeout: int = 5):
        API.__init__(self, default_timeout=default_timeout)
        GrammarVisitor.__init__(self, grammar_request)
        self.path = None
        self.mode = None
        self.alias = False
        self.override = False

    def visit_options(self, node: DerivationTree):
        self.path = None
        self.mode = None
        self.alias = False
        self.override = False
        self.generic_visit(node)

    def visit_alias(self, node: DerivationTree):
        self.alias = True

    def visit_override(self, node: DerivationTree):
        self.override = True

    def visit_url(self, node: DerivationTree):
        self.path = node.children[1].to_string()

    def visit_mode(self, node: DerivationTree):
        self.mode = node.children[1].to_string()

    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return False

    def fallback_condition(self, process: subprocess.CompletedProcess) -> bool:
        return False

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        return False

    def fallback_contains(self, process: subprocess.CompletedProcess) -> bool:
        return False

    # noinspection PyBroadException
    def run(self, system_test_path: PathLike, environ: Environment) -> TestResult:
        try:
            with open(system_test_path, "r") as fp:
                test = fp.read()
            if test:
                self.visit_source(test)
                test = test.split("\n")
            else:
                test = []
            process = subprocess.run(
                ["python", HARNESS_FILE] + test,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.default_timeout,
                env=environ,
            )
            if self.condition(process) and self.contains(process):
                return TestResult.FAILING
            else:
                if self.fallback_condition(process) and self.fallback_contains(process):
                    return TestResult.FAILING
                elif self.error_handling(process):
                    print(process)
                    return TestResult.UNDEFINED
                else:
                    return TestResult.PASSING
        except subprocess.TimeoutExpired:
            return TestResult.UNDEFINED
        except Exception as e:
            traceback.print_exception(e)
            return TestResult.UNDEFINED

    def error_handling(self, process) -> bool:
        return process.returncode != 0 and process.returncode != 200


class FastAPI2API(FastAPIDefaultAPI):
    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return self.mode == "websocket" and self.path == "/router/" and self.override

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        return b"Overridden" not in process.stdout

    def fallback_condition(self, process: subprocess.CompletedProcess) -> bool:
        return (
            self.mode == "websocket" and self.path == "/router/" and not self.override
        )

    def fallback_contains(self, process: subprocess.CompletedProcess) -> bool:
        return b"Dependency" not in process.stdout


class FastAPI3API(FastAPIDefaultAPI):
    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return process.returncode != 0 and process.returncode != 200

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        return (
            b"pydantic.error_wrappers.ValidationError:" in process.stderr
            and (
                b"validation errors for Item" in process.stderr
                or b"validation error for Item" in process.stderr
                or b"validation error for OtherItem" in process.stderr
                or b"validation errors for OtherItem" in process.stderr
            )
            and b"aliased_name" in process.stderr
            and b"field required (type=value_error.missing)" in process.stderr
        )


class FastAPI4API(FastAPIDefaultAPI):
    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return process.returncode in (0, 200) and self.path == "/openapi.json"

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        response = eval(process.stdout.decode("utf-8"))
        if not isinstance(response, dict):
            return False
        key_queue = queue.Queue()
        for key in response:
            key_queue.put((key, response[key]))
        while not key_queue.empty():
            key, value = key_queue.get()
            if key == "parameters":
                if isinstance(value, list):
                    result = sorted(list(map(str, value)))
                    expected = [*set(result)]
                    if result != expected:
                        return True
            elif isinstance(value, dict):
                for k in value:
                    key_queue.put((k, value[k]))
        return False


class FastAPI5API(FastAPIDefaultAPI):
    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return process.returncode == 0 or process.returncode == 200

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        return (
            b'"password"' in process.stdout
            or b'"test-password"' in process.stdout
            or b"'password'" in process.stdout
            or b"'test-password'" in process.stdout
        )


class FastAPI6API(FastAPIDefaultAPI):
    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return process.returncode == 166

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        return b"value_error.missing" in process.stdout


class FastAPI7API(FastAPIDefaultAPI):
    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return process.returncode != 0 and process.returncode != 200

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        return (
            b"TypeError: Object of type Decimal is not JSON serializable"
            in process.stderr
        )

    def error_handling(self, process: subprocess.CompletedProcess) -> bool:
        return (
            process.returncode != 0
            and process.returncode != 200
            and (
                process.returncode != 166
                or b"value_error.number.not_gt" not in process.stdout
            )
        )


class FastAPI8API(FastAPIDefaultAPI):
    def condition(self, process: subprocess.CompletedProcess) -> bool:
        return (
            self.path == "/routes/"
            and process.returncode != 0
            and process.returncode != 200
        )

    def contains(self, process: subprocess.CompletedProcess) -> bool:
        return (
            b"AttributeError: 'APIRoute' object has no attribute 'x_type'"
            in process.stderr
        )


grammar_jsonable_encoder: Grammar = {
    "<start>": ["<options>"],
    "<options>": ["", "<option_list>"],
    "<option_list>": ["<option>", "<option_list>\n<option>"],
    "<option>": [
        "<obj>",
        "<include>",
        "<exclude>",
        "<by_alias>",
        "<skip_defaults>",
        "<exclude_unset>",
        "<exclude_defaults>",
        "<include_none>",
        "<exclude_none>",
        "<custom_encoder>",
        "<sqlalchemy_safe>",
    ],
    # OPTIONS
    "<obj>": ["-o<object>"],
    "<include>": ["-i<key_list>"],
    "<exclude>": ["-e<key_list>"],
    "<by_alias>": ["-a"],
    "<skip_defaults>": ["-s"],
    "<exclude_unset>": ["-u"],
    "<exclude_defaults>": ["-d"],
    "<include_none>": ["-ni"],
    "<exclude_none>": ["-ne"],
    "<custom_encoder>": ["-c<custom_encoders>"],
    "<sqlalchemy_safe>": ["-q"],
    # MODEL
    "<object>": ["<dict>", "<model>"],
    "<dict>": ["{}", "{<dict_entries>}"],
    "<dict_entries>": ["<dict_entry>", "<dict_entries>,<dict_entry>"],
    "<dict_entry>": ["'<key>':<str>"],
    "<model>": ["Model()", "Model(<parameters>)"],
    "<parameters>": ["<parameter>", "<parameters>,<parameter>"],
    "<parameter>": ["<key>=<str>"],
    # LISTS
    "<key_list>": ["<key>", "<key_list>,<key>"],
    # ENCODERS
    "<custom_encoders>": ["{str:repr}"],
    # UTILS
    "<key>": ["foo", "bar", "bla", "da"],
    "<str>": ["''", "'<chars>'"],
    "<chars>": ["<char>", "<chars><char>"],
    "<char>": srange(string.ascii_letters + string.digits + "_ "),
}

assert is_valid_grammar(grammar_jsonable_encoder)

grammar_request: Grammar = {
    "<start>": ["<options>"],
    "<options>": ["", "<option_list>"],
    "<option_list>": ["<option>", "<option_list>\n<option>"],
    "<option>": [
        "<url>",
        "<mode>",
        "<data>",
        "<alias>",
        "<override>",
        "<users>",
    ],
    # OPTIONS
    "<url>": ["-p<path>"],
    "<mode>": ["-m<r_mode>"],
    "<data>": ["-d<json>"],
    "<alias>": ["-a"],
    "<override>": ["-o"],
    "<users>": ["-u"],
    # UTILS
    "<path>": ["/<chars>", "/<chars>/", "/<chars><path>"],
    "<r_mode>": ["get", "post", "websocket"],
    "<json>": ["<json_object>", "<json_list>", "<json_value>"],
    "<json_object>": ["{}", "{<pairs>}"],
    "<pairs>": ["<pair>", "<pairs>,<pair>"],
    "<pair>": ["<key>:<json_value>"],
    "<json_list>": ["[]", "[<json_values>]"],
    "<json_values>": ["<json_value>", "<json_values>,<json_value>"],
    "<json_value>": ["<number>", "<str>", "<json_object>", "<json_list>"],
    "<key>": ["<str>"],
    "<str>": ['""', '"<chars>"'],
    "<chars>": ["<char>", "<chars><char>"],
    "<char>": srange(string.ascii_letters + string.digits + "_-. "),
    "<number>": ["<int>", "<float>"],
    "<int>": ["<nonzero><digits>", "-<nonzero><digits>", "0", "-0"],
    "<digit>": srange(string.digits),
    "<digits>": ["", "<digits><digit>"],
    "<nonzero>": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<float>": ["<int>.<digit><digits>"],
}

assert is_valid_grammar(grammar_request)
