import string
from fuzzingbook.Grammars import srange, Grammar
from fuzzingbook.Parser import EarleyParser, tree_to_string


HEARTBLEED: Grammar = {
    "<start>": ["<length> <payload> <padding>"],
    "<length>": ["<onenine><maybe_digits>"],
    "<onenine>": srange("123456789"),
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<digit>": list(string.digits),
    "<payload>": ["p", "p<string>"],
    "<padding>": ["x", "x<string>"],
    "<string>": ["<char>", "<char><string>"],
    "<char>": list(string.ascii_letters) + list(""),
}


INITIAL_INPUTS = ["3 pab x", "3 pxy xpadding", "8 pasbd xyasd"]


def prop(inp):
    s = str(inp).split()
    length = int(s[0])
    payload_length = len(s[1])
    if length > int(payload_length):
        return True
    return False


if __name__ == "__main__":
    p = EarleyParser(HEARTBLEED)

    for inp in INITIAL_INPUTS:
        for tree in p.parse(inp):
            assert len(tree) != 0
            assert tree_to_string(tree) == inp
