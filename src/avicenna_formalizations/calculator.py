import string
import math

from fuzzingbook.Grammars import Grammar
from avicenna.input import Input


grammar: Grammar = {
    "<start>": ["<arith_expr>"],
    "<arith_expr>": ["<function>(<number>)"],
    "<function>": ["sqrt", "sin", "cos", "tan"],
    "<number>": ["<maybe_minus><onenine><maybe_digits><maybe_frac>"],
    "<maybe_minus>": ["", "-"],
    "<onenine>": [str(num) for num in range(1, 10)],
    "<digit>": list(string.digits),
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<maybe_frac>": ["", ".<digits>"],
}


initial_inputs = ["cos(10)", "sqrt(28367)", "tan(-12)", "sqrt(-900)"]


def arith_eval(inp: str) -> float:
    return eval(
        str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
    )


def prop(inp: Input) -> bool:
    try:
        arith_eval(str(inp))
        return False
    except ValueError:
        return True
