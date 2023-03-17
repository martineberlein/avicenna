import string
from fuzzingbook.Grammars import srange, Grammar
from fuzzingbook.Parser import EarleyParser, tree_to_string

from avicenna.input import Input
from avicenna.oracle import OracleResult


grammar: Grammar = {
    "<start>": ["<length> <payload> <padding>"],
    "<length>": ["<one_nine><maybe_digits>"],
    "<one_nine>": srange("123456789"),
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<digit>": list(string.digits),
    "<payload>": ["<string>"],
    "<padding>": ["<string>"],
    "<string>": ["<char>", "<char><string>"],
    "<char>": list(string.ascii_letters),
}


initial_inputs = ["3 pab x", "3 pxy xpadding", "8 pasbd xyasd"]


def prop_(inp):
    s = str(inp).split()
    length = int(s[0])
    payload_length = len(s[1])
    if length > int(payload_length):
        return True
    return False


def prop(test_input: Input):
    s = str(test_input).split()
    length = int(s[0])
    payload_length = len(s[1])
    if length > int(payload_length):
        return OracleResult.BUG
    return OracleResult.NO_BUG


if __name__ == "__main__":
    p = EarleyParser(grammar=grammar)

    for inp in initial_inputs:
        for tree in p.parse(inp):
            assert len(tree) != 0
            assert tree_to_string(tree) == inp
