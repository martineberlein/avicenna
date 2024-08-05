from fuzzingbook.Grammars import Grammar, is_valid_grammar

from debugging_framework.input.oracle import OracleResult
from avicenna.input.input import Input


grammar: Grammar = {
    "<start>": ["<x>, <y>, <z>"],
    "<x>": ["<integer>"],
    "<y>": ["<integer>"],
    "<z>": ["<integer>"],
    "<integer>": ["<digit>", "<digit><integer>"],
    "<digit>": [str(num) for num in range(1, 10)],
}
assert is_valid_grammar(grammar)


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
            return OracleResult.PASSING
        else:
            return OracleResult.FAILING
    except BaseException:
        return OracleResult.FAILING


def _test_middle(x, y, z, expected):
    return _test(middle, x, y, z, expected)


def _test_dummy(inp: str) -> OracleResult:
    def oracle(x, y, z):
        sorted_list = sorted([x, y, z])
        return sorted_list[1]

    x, y, z = eval(str(inp))
    truth = oracle(x, y, z)
    return _test_middle(x, y, z, expected=truth)


def oracle(inp: Input | str) -> OracleResult:
    return _test_dummy(str(inp))


initial_inputs = ["3, 1, 4", "3, 2, 1"]


if __name__ == "__main__":
    assert _test_dummy("3, 1, 4") == OracleResult.FAILING
    assert _test_dummy("3, 2, 1") == OracleResult.PASSING
