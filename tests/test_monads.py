import unittest
from typing import Set, Iterable, Type

from avicenna.oracle import OracleResult
from avicenna.input import Input
from avicenna.monads import Maybe, Just, Nothing, Exceptional, Success, Failure, T, E
from avicenna_formalizations.calculator import grammar, oracle
from avicenna.feature_collector import GrammarFeatureCollector
from avicenna.execution_handler import SingleExecutionHandler


class MyTestCase(unittest.TestCase):
    def test_monads(self):
        inputs = [
            ("sqrt(-901)", OracleResult.BUG),
            ("sqrt(-1)", OracleResult.BUG),
            ("sqrt(10)", OracleResult.NO_BUG),
            ("cos(1)", OracleResult.NO_BUG),
            ("sin(99)", OracleResult.NO_BUG),
            ("tan(-20)", OracleResult.NO_BUG),
        ]

        result: Maybe = (
            Maybe.just(inputs)
            .map(parse_to_input)
            .map(assign_label)
            .map(assign_feature_vector)
        )

        if result.is_just():
            print(result.value())
            for inp in result.value():
                print(inp, inp.oracle, inp.features)

    def test_success_failure_monad(self):
        inputs = [
            ("sqrt(-901)", OracleResult.BUG),
            ("sqrt(-1)", OracleResult.BUG),
            ("sqrt(10)", OracleResult.NO_BUG),
            ("cos(1)", OracleResult.NO_BUG),
            ("sin(99)", OracleResult.NO_BUG),
            ("tan(-20)", OracleResult.NO_BUG),
        ]
        # inputs = []
        parsed_inputs: Set[Input] = (
            Exceptional.of(lambda: inputs)
            .map(parse_to_input)
            .map(assign_label)
            .map(assign_feature_vector)
            .reraise()
            .map(lambda x: x if x else None)
            .get()
        )

        if parsed_inputs:
            for inp in parsed_inputs:
                print(inp, inp.oracle, inp.features)
        else:
            print("Empty Set was returned")

    def test_success_failure_monad(self):
        inputs = [
            ("sqrt(-901)", OracleResult.BUG),
            ("sqrt(-1)", OracleResult.BUG),
            ("sqrt(10)", OracleResult.NO_BUG),
            ("cos(1)", OracleResult.NO_BUG),
            ("sin(99)", OracleResult.NO_BUG),
            ("tan(-20)", OracleResult.NO_BUG),
        ]
        parsed_inputs: Set[Input] = (
            Exceptional.of(lambda: None)
            .map(parse_to_input)
            .map(assign_label)
            .map(assign_feature_vector)
            .recover(lambda _: None, SyntaxError)
            .bind(check_empty)
            .recover(lambda _: {Input.from_str(grammar, inp_) for inp_ in ["sqrt(-1)"]})
            .get()
        )

        if parsed_inputs:
            for inp in parsed_inputs:
                print(inp, inp.oracle, inp.features)
        else:
            print("Empty Set was returned")


def check_empty(x: T) -> Exceptional[Exception, T]:
    if x is None:
        return Failure(AssertionError())
    elif isinstance(x, (str, list, dict, set, tuple)) and not x:
        return Failure(AssertionError())
    return Success(x)

def parse_to_input(test_inputs: Iterable[str]) -> Set[Input]:
    return set([Input.from_str(grammar, inp_) for inp_, _ in test_inputs])

def assign_label(test_inputs: Set[Input]) -> Set[Input]:
    execution_handler = SingleExecutionHandler(oracle)
    execution_handler.label(test_inputs)
    return test_inputs

def assign_feature_vector(test_inputs: Set[Input]) -> Set[Input]:
    collector = GrammarFeatureCollector(grammar)
    for inp_ in test_inputs:
        inp_.features = collector.collect_features(inp_)
    return test_inputs


if __name__ == "__main__":
    unittest.main()
