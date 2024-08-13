import string

from avicenna import Avicenna
from avicenna.data import OracleResult

import evaluation.resources.seed
from evaluation.resources.output import print_diagnoses


# Oracle for divide by zero
def divide_by_zero_oracle(inp: str):
    try:
        eval(str(inp))
    except ZeroDivisionError as e:
        return OracleResult.FAILING, ZeroDivisionError
    return OracleResult.PASSING, None


# Initial inputs for Avicenna
divide_by_zero_inputs = ["1/0", "5/(3-3)", "(2+3)/5", "7/(2*0)", "9/(0/3)"]

divide_by_zero_grammar = {
    "<start>": ["<arith_expr>"],
    "<arith_expr>": [
        "<arith_expr><operator><arith_expr>",
        "<number>",
        "(<arith_expr>)",
    ],
    "<operator>": ["+", "-", "*", "/"],
    "<number>": [
        "<maybe_minus><non_zero_digit><maybe_digits><maybe_frac>",
        "<maybe_minus>0.<digits>",
        "<maybe_minus>0",
    ],
    "<maybe_minus>": ["", "-"],
    "<non_zero_digit>": [
        str(num) for num in range(1, 10)
    ],  # Exclude 0 from starting digits
    "<digit>": list(string.digits),
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<maybe_frac>": ["", ".<digits>"],
}


if __name__ == "__main__":
    param = {
        "max_iterations": 5,
        "grammar": divide_by_zero_grammar,
        "initial_inputs": divide_by_zero_inputs,
        "oracle": divide_by_zero_oracle,
    }

    avicenna = Avicenna(**param, enable_logging=True)
    diagnoses = avicenna.explain()
    print_diagnoses(diagnoses)
