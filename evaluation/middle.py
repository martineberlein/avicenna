from enum import Enum
import logging

from fuzzingbook.Grammars import Grammar, is_valid_grammar
from avicenna.oracle import OracleResult
from avicenna.input import Input
from isla.language import ISLaUnparser


class TestResult(Enum):
    BUG = "BUG"
    NO_BUG = "NO_BUG"

    def __repr__(self):
        return self.value


def middle(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = y
        elif x < z:
            m = y  # bug
    else:
        if x > y:
            m = y
        elif x > z:
            m = x
    return m


def _test(function, x, y, z, expected):
    try:
        if function(x, y, z) == expected:
            return TestResult.NO_BUG
        else:
            return TestResult.BUG
    except BaseException:
        return TestResult.BUG


def _test_middle(x, y, z, expected):
    return _test(middle, x, y, z, expected)


def _test_dummy(inp: str) -> OracleResult:
    def oracle(x, y, z):
        sorted_list = sorted([x, y, z])
        return sorted_list[1]

    x, y , z = eval(str(inp))
    truth = oracle(x,y,z)
    return OracleResult.BUG if _test_middle(x,y,z, expected=truth) == TestResult.BUG else OracleResult.NO_BUG


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")

    assert _test_dummy("3, 1, 4") == OracleResult.BUG
    assert _test_dummy("3, 2, 1") == OracleResult.NO_BUG

    grammar: Grammar = {
        "<start>": ['<x>, <y>, <z>'],
        "<x>": ["<integer>"],
        "<y>": ["<integer>"],
        "<z>": ["<integer>"],
        "<integer>": ["<digit>", "<digit><integer>"],
        "<digit>": [str(num) for num in range(1, 10)]
    }
    assert is_valid_grammar(grammar)

    from avicenna import Avicenna

    def prop(inp) -> bool:
        return True if _test_dummy(inp) == OracleResult.BUG else False

    def oracle(inp: Input) -> OracleResult:
        return _test_dummy(str(inp))

    avicenna = Avicenna(
        grammar=grammar,
        initial_inputs=["3, 1, 4", "3, 2, 1"],
        oracle=oracle,
        max_excluded_features=4,
        max_iterations=10
    )

    result = avicenna.explain()
