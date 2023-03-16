from typing import Dict, List, Tuple, Collection
import subprocess
import enum
import os
from abc import abstractmethod
from os import PathLike
from pathlib import Path
import re
from abc import ABC
from typing import Callable, Any

from fuzzingbook.Grammars import Grammar
from isla.derivation_tree import DerivationTree
from isla.parser import EarleyParser


# ~~~~~~ CONSTANTS ~~~~~~ #

T4P_DIR = Path(Path(__file__).parent.parent.parent.parent / 'Tests4Py').absolute()

# ~~~~~~ TYPES ~~~~~~ #

Environment = Dict[str, str]

# ~~~~~~ FILES ~~~~~~ #

HARNESS_FILE = "harness_fastapi_1.py"



class TestResult(enum.Enum):
    FAILING = 0
    PASSING = 1
    UNDEFINED = 2


class API:
    def __init__(self, default_timeout=5):
        self.default_timeout = default_timeout

    @abstractmethod
    def run(self, system_test_path: PathLike, environ: Environment) -> TestResult:
        return NotImplemented

    def runs(
        self, system_tests_path: PathLike, environ: Environment
    ) -> List[Tuple[PathLike, TestResult]]:
        system_tests_path = Path(system_tests_path)
        if not system_tests_path.exists():
            raise ValueError(f"{system_tests_path} does not exist")
        if not system_tests_path.is_dir():
            raise ValueError(f"{system_tests_path} is not a directory")
        tests = list()
        for dir_path, _, files in os.walk(system_tests_path):
            for file in files:
                path = Path(dir_path, file)
                tests.append((path, self.run(path, environ)))
        return tests


class ExpectOutputAPI(API):
    def __init__(
        self,
        expected: bytes | Collection[bytes],
        executable: PathLike = HARNESS_FILE,
        expect_in: bool = False,
        is_stdout: bool = False,
        no_check: bool = False,
        is_or: bool = True,
        default_timeout: int = 5,
    ):
        self.expected = expected
        self.executable = executable
        self.is_stdout = is_stdout
        self.expect_in = expect_in
        self.no_check = no_check
        self.is_or = is_or
        super().__init__(default_timeout=default_timeout)

    # noinspection PyBroadException
    def run(self, system_test_path: PathLike, environ: Environment) -> TestResult:
        try:
            with open(system_test_path, "r") as fp:
                test = fp.read()
            if test:
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
            if self.no_check or process.returncode:
                if (any if self.is_or else all)(
                    map(
                        (
                            process.stdout if self.is_stdout else process.stderr
                        ).__contains__,
                        [self.expected]
                        if isinstance(self.expected, bytes)
                        else self.expected,
                    )
                ):
                    return TestResult.PASSING if self.expect_in else TestResult.FAILING
                elif self.no_check:
                    return TestResult.FAILING if self.expect_in else TestResult.PASSING
                else:
                    return TestResult.UNDEFINED
            else:
                return TestResult.PASSING
        except subprocess.TimeoutExpired:
            return TestResult.UNDEFINED
        except Exception:
            return TestResult.UNDEFINED




class ExpectErrAPI(ExpectOutputAPI):
    def __init__(
        self, expected: bytes | Collection[bytes], executable: PathLike = HARNESS_FILE
    ):
        super().__init__(expected, executable)


class ExpectOutAPI(ExpectOutputAPI):
    def __init__(
        self, expected: bytes | Collection[bytes], executable: PathLike = HARNESS_FILE
    ):
        super().__init__(expected, executable, is_stdout=True)


class ExpectNotErrAPI(ExpectOutputAPI):
    def __init__(
        self, expected: bytes | Collection[bytes], executable: PathLike = HARNESS_FILE
    ):
        super().__init__(expected, executable, expect_in=True)


class ExpectNotOutAPI(ExpectOutputAPI):
    def __init__(
        self, expected: bytes | Collection[bytes], executable: PathLike = HARNESS_FILE
    ):
        super().__init__(expected, executable, is_stdout=True, expect_in=True)



ILLEGAL_CHARS = re.compile(r"[^A-Za-z0-9_]")


class GrammarVisitor(ABC):
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.parser = EarleyParser(self.grammar)

    def visit_source(self, source: str):
        for tree in self.parser.parse(source):
            return self.visit(DerivationTree.from_parse_tree(tree))
        else:
            raise SyntaxError(
                f'"{source}" is not parsable with the grammar {self.grammar}'
            )

    @staticmethod
    def get_name(value: str):
        return ILLEGAL_CHARS.sub("", value)

    def generic_visit(self, node: DerivationTree) -> Any:
        for child in node.children:
            self.visit(child)

    def visit(self, node: DerivationTree) -> Any:
        method = "visit_" + self.get_name(node.value)
        visitor: Callable[[DerivationTree], None] = getattr(
            self, method, self.generic_visit
        )
        return visitor(node)


class Generator:
    def generate(self):
        NotImplemented()

    def reset(self):
        NotImplemented()
