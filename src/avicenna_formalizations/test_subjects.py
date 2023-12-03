import importlib.util
import sys
from ast import literal_eval
from typing import Union, Type, List, Callable, Dict, Tuple, Set
from pathlib import Path
from abc import ABC
import string
from fuzzingbook.Coverage import Coverage, Location, BranchCoverage

from fuzzingbook.Grammars import Grammar
from avicenna.oracle import OracleResult
from avicenna.oracle_construction import construct_oracle


class TestSubject:
    def __init__(self, grammar=None, oracle=None, test_inputs=None):
        self.grammar = grammar or self.default_grammar
        self.oracle = oracle or self.default_oracle()
        self.test_inputs = test_inputs or self.default_test_inputs

    default_grammar = {}
    default_test_inputs = []

    def default_oracle(self):
        raise ValueError("Default Oracle not implemented")

    def to_dict(self):
        return {
            "grammar": self.grammar,
            "oracle": self.oracle,
            "initial_inputs": self.test_inputs,
        }


class RefactoryTestSubject(TestSubject):
    base_path: Path
    implementation_function_name: str

    def __init__(self, oracle, bug_id, solution_type: str):
        """
        Oracle allways needs to be constructed!

        :param oracle: constructed Differential Oracle
        :param bug_id: number of the implementation XYZ_N_001, XYZ_N_002 etc.
        :param solution_type: can be either 'correct', 'fail', or 'wrong'
        """
        super().__init__(oracle=oracle)
        self.bug_id = bug_id
        self.solution_type = solution_type

    @classmethod
    def harness_function(cls, input_str: str):
        raise NotImplementedError

    @classmethod
    def ground_truth(cls) -> Callable:
        solution_file_path = cls.base_path / Path("reference/reference.py")
        return load_object_dynamically(
            solution_file_path,
            cls.implementation_function_name,
        )

    def get_implementation(self) -> Callable:
        imp_dir_path = self.base_path / Path(self.solution_type)
        imp_file_path = list(imp_dir_path.absolute().glob(f"*{self.bug_id}.py"))[0]

        func = load_object_dynamically(
            imp_file_path,
            self.implementation_function_name,
        )
        return func


class Question1RefactoryTestSubject(RefactoryTestSubject):
    base_path = Path("./resources/refactory/question_1/code")
    implementation_function_name = "search"
    default_grammar: Grammar = {
        "<start>": ["<input>"],
        "<input>": ["<first>, <second>"],
        "<first>": ["<integer>"],
        "<second>": ["()", "[]", "(<integer>, <integer><list>)", "[<integer><list>]"],
        "<list>": ["", ", <integer><list>"],
        "<integer>": ["<maybe_minus><one_nine><maybe_digits>"],
        "<maybe_minus>": ["", "-"],  #
        "<one_nine>": [str(num) for num in range(1, 10)],
        "<digit>": list(string.digits),
        "<maybe_digits>": ["", "<digits>"],
        "<digits>": ["<digit>", "<digit><digits>"],
    }
    default_test_inputs = ["42, (-5, 1, 3, 5, 7, 10)", "3, (1, 5, 10)"]

    @classmethod
    def harness_function(cls, input_str: str):
        # Split the string into two parts based on the first comma and a space
        arg1_str, arg2_str = input_str.split(", ", 1)

        # Convert the string parts to Python literals
        arg1 = literal_eval(arg1_str)
        arg2 = literal_eval(arg2_str)

        return arg1, arg2


class RefactoryTestSubjectFactory:
    def __init__(self, test_subject_type: Type[RefactoryTestSubject]):
        self.test_subject_type = test_subject_type

    def build(
        self,
        err_def: Dict[Exception, OracleResult] = None,
        default_oracle: OracleResult = None,
        solution_type: str = "wrong",
    ) -> List[RefactoryTestSubject]:
        subjects = []

        subject_path = Path(self.test_subject_type.base_path) / Path(solution_type)
        num_files = len(list(subject_path.absolute().glob(f"*.py")))

        for i in range(1, num_files):
            formatted_str = str(i).zfill(3)

            try:
                subject = self._build_subject(formatted_str, err_def, default_oracle, solution_type)
                subjects.append(subject)
            except Exception as e:
                print(f"Subject {formatted_str} could not be build.")

        return subjects

    def _build_subject(
        self,
        formatted_bug_id: str,
        err_def: Dict[Exception, OracleResult] = None,
        default_oracle: OracleResult = None,
        solution_type: str = "wrong",
    ):

        reference = self.test_subject_type.ground_truth()
        subject = self.test_subject_type(
            oracle=lambda _: None, bug_id=formatted_bug_id, solution_type=solution_type
        )
        implementation = subject.get_implementation()

        error_def = err_def or {TimeoutError: OracleResult.UNDEF}
        def_oracle = default_oracle or OracleResult.BUG

        oracle = construct_oracle(
            implementation,
            reference,
            error_def,
            default_oracle_result=def_oracle,
            timeout=0.01,
            harness_function=subject.harness_function,
        )
        subject.oracle = oracle

        return subject


class MPITestSubject(TestSubject, ABC):
    base_path: str
    implementation_class_name: str = "Solution"
    implementation_function_name: str

    def __init__(self, oracle, bug_id):
        super().__init__(oracle=oracle)
        self.bug_id = bug_id

    @classmethod
    def ground_truth(cls) -> Callable:
        solution_file_path = cls.base_path / Path("reference1.py")
        return load_function_from_class(
            solution_file_path,
            cls.implementation_class_name,
            cls.implementation_function_name,
        )

    def get_implementation(self) -> Callable:
        imp_file_path = self.base_path / Path(f"prog_{self.bug_id}/buggy.py")

        func = load_function_from_class(
            imp_file_path,
            self.implementation_class_name,
            self.implementation_function_name,
        )

        def harness_function(inp: str):
            param = list(map(int, str(inp).strip().split()))
            return func(*param)

        return harness_function


class GCDTestSubject(MPITestSubject):
    base_path = Path("./resources/mpi/problem_1_GCD")
    implementation_function_name = "gcd"
    default_grammar: Grammar = {
        "<start>": ["<input>"],
        "<input>": ["<first> <second>"],
        "<first>": ["<integer>"],
        "<second>": ["<integer>"],
        "<integer>": ["<one_nine><maybe_digits>"],
        "<one_nine>": [str(num) for num in range(1, 10)],
        "<digit>": list(string.digits),
        "<maybe_digits>": ["", "<digits>"],
        "<digits>": ["<digit>", "<digit><digits>"],
    }
    default_test_inputs = ["10 2", "4 4"]


class SquareRootTestSubject(MPITestSubject):
    base_path = Path("./resources/mpi/problem_10_Square-root")
    implementation_function_name = "floorSqrt"
    default_grammar: Grammar = {
        "<start>": ["<input>"],
        "<input>": ["<integer>"],
        "<integer>": ["<one_nine><maybe_digits>", "0"],
        "<one_nine>": [str(num) for num in range(1, 10)],
        "<digit>": list(string.digits),
        "<maybe_digits>": ["", "<digits>"],
        "<digits>": ["<digit>", "<digit><digits>"],
    }
    default_test_inputs = ["4", "5"]


class MiddleTestSubject(MPITestSubject):
    base_path = Path("./resources/mpi/problem_7_Middle-of-Three")
    implementation_function_name = "middle"
    default_grammar: Grammar = {
        "<start>": ["<input>"],
        "<input>": ["<first> <second> <third>"],
        "<first>": ["<integer>"],
        "<second>": ["<integer>"],
        "<third>": ["<integer>"],
        "<integer>": ["<one_nine><maybe_digits>"],
        "<one_nine>": [str(num) for num in range(1, 10)],
        "<digit>": list(string.digits),
        "<maybe_digits>": ["", "<digits>"],
        "<digits>": ["<digit>", "<digit><digits>"],
    }
    default_test_inputs = ["978 518 300", "162 934 200"]


class MPITestSubjectFactory:
    def __init__(self, test_subject_type: Type[MPITestSubject]):
        self.test_subject_type = test_subject_type

    def build(
        self,
        err_def: Dict[Exception, OracleResult] = None,
        default_oracle: OracleResult = None,
    ) -> List[MPITestSubject]:
        subjects = []
        for i in range(1, 11):
            buggy_file_path = self.test_subject_type.base_path / Path(
                f"prog_{i}/buggy.py"
            )
            loaded_function = load_function_from_class(
                buggy_file_path,
                self.test_subject_type.implementation_class_name,
                self.test_subject_type.implementation_function_name,
            )
            error_def = err_def or {TimeoutError: OracleResult.UNDEF}
            def_oracle = default_oracle or OracleResult.BUG
            oracle = construct_oracle(
                loaded_function,
                self.test_subject_type.ground_truth(),
                error_def,
                default_oracle_result=def_oracle,
                timeout=0.01,
            )
            subjects.append(self.test_subject_type(oracle=oracle, bug_id=i))
        return subjects


def load_module_dynamically(path: Union[str, Path]):
    # Step 1: Convert file path to module name
    file_path = str(path.absolute())
    module_name = file_path.replace("/", ".").rstrip(".py")

    # Step 2: Load module dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_object_dynamically(path: Union[str, Path], object_name: str):
    module = load_module_dynamically(path)
    return getattr(module, object_name)


def load_function_from_class(
    path: Union[str, Path], class_name: str, function_name: str
):
    class_ = load_object_dynamically(path, class_name)
    function = getattr(class_(), function_name)

    return function


def population_coverage(
    population: List[Tuple[int, int]], function: Callable
) -> Tuple[Set[Location], List[int]]:
    cumulative_coverage: List[int] = []
    all_coverage: Set[Location] = set()

    for s in population:
        with Coverage() as cov:
            try:
                function(s)
            except:
                pass
        filtered_set = {
            (func, line)
            for (func, line) in cov.coverage()
            if "derivation_tree" not in func and "input" not in func
        }
        all_coverage |= filtered_set
        cumulative_coverage.append(len(all_coverage))

    return all_coverage, cumulative_coverage


def population_branch_coverage(
    population: List[Tuple[int, int]], function: Callable
) -> Tuple[Set[Location], List[int]]:
    cumulative_coverage: List[int] = []
    all_coverage: Set[Location] = set()

    for s in population:
        with BranchCoverage() as cov:
            try:
                function(s)
            except:
                pass
        filtered_set = {
            (x, y)
            for (x, y) in cov.coverage()
            if "derivation_tree" not in x[0] and y[0] and "input" not in x[0] and y[0]
        }
        all_coverage |= filtered_set
        cumulative_coverage.append(len(all_coverage))

    return all_coverage, cumulative_coverage


def main():
    subjects = MPITestSubjectFactory(MiddleTestSubject).build()
    for subject in subjects:
        param = subject.to_dict()
        orc = subject.get_implementation()
        print(population_coverage(param.get("initial_inputs"), orc))


def main2():

    sub_types = [
        Question1RefactoryTestSubject,
    ]

    factory = RefactoryTestSubjectFactory(Question1RefactoryTestSubject)
    subjects = factory.build(solution_type="fail")
    for subject in subjects:
        print(subject.bug_id)
        orc = subject.to_dict().get("oracle")
        inputs = subject.to_dict().get("initial_inputs")
        for inp in inputs:
            print(orc(inp))


def main3():
    from fuzzingbook.GrammarFuzzer import GrammarFuzzer

    fuzzer = GrammarFuzzer(Question1RefactoryTestSubject.default_grammar)
    for _ in range(30):
        print(fuzzer.fuzz())


if __name__ == "__main__":
    main2()
