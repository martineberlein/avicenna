from fuzzingbook.Grammars import (
    crange,
    srange,
    Grammar,
    CHARACTERS_WITHOUT_QUOTE,
    convert_ebnf_grammar,
)
from fuzzingbook.Parser import EarleyParser
from isla.language import DerivationTree

JSON_EBNF_GRAMMAR: Grammar = {
    "<start>": ["<json>"],
    "<json>": ["<element>"],
    "<element>": ["<ws><value><ws>"],
    "<value>": ["<object>", "<array>", "<string>", "<number>", "true", "false", "null"],
    "<object>": ["{<ws>}", "{<members>}"],
    "<members>": ["<member>(,<members>)*"],
    "<member>": ["<ws><string><ws>:<element>"],
    "<array>": ["[<ws>]", "[<elements>]"],
    "<elements>": ["<element>(,<elements>)*"],
    "<string>": ['"' + "<characters>" + '"'],
    "<characters>": ["<character>*"],
    "<character>": srange(CHARACTERS_WITHOUT_QUOTE),
    "<number>": ["<int><frac><exp>"],
    "<int>": ["<digit>", "<onenine><digits>", "-<digit>", "-<onenine><digits>"],
    "<digits>": ["<digit>+"],
    "<digit>": ["0", "<onenine>"],
    "<onenine>": crange("1", "9"),
    "<frac>": ["", ".<digits>"],
    "<exp>": ["", "E<sign><digits>", "e<sign><digits>"],
    "<sign>": ["", "+", "-"],
    # "<ws>": srange(string.whitespace)
    "<ws>": [""],
}

JSON_GRAMMAR = convert_ebnf_grammar(JSON_EBNF_GRAMMAR)


def prop(inp: DerivationTree):
    def iter_tree(derivation_tree):
        n, c = derivation_tree
        if n == "<object>":
            return True
        else:
            result = False
            for child in c:
                result = iter_tree(child)
                if result:
                    return result
            return result

    if isinstance(inp, DerivationTree):
        return iter_tree(inp)
    else:
        parser = EarleyParser(JSON_GRAMMAR)
        for derivation_tree in parser.parse(str(inp)):
            return iter_tree(derivation_tree)


INITIAL_INPUTS = ["[]", '[{"U":-50.9E-05023},false,false]']
