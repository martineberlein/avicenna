import string
from isla.language import ISLaUnparser

from avicenna import Avicenna
from avicenna.input import OracleResult
from avicenna.pattern_learner import AvicennaPatternLearner


# Oracle for divide by zero
def divide_by_zero_oracle(inp: str):
    try:
        eval(str(inp))
    except ZeroDivisionError as e:
        return OracleResult.FAILING, ZeroDivisionError
    return OracleResult.PASSING, None


# Initial inputs for Avicenna
divide_by_zero_inputs = ['1/0', '5/(3-3)', '(2+3)/5', '7/(2*0)', '9/(0/3)']

divide_by_zero_grammar = {
    "<start>": ["<arith_expr>"],
    "<arith_expr>": ["<arith_expr><operator><arith_expr>", "<number>", "(<arith_expr>)"],
    "<operator>": ["+", "-", "*", "/"],
    "<number>": ["<maybe_minus><non_zero_digit><maybe_digits><maybe_frac>", "<maybe_minus>0.<digits>",
                 "<maybe_minus>0"],
    "<maybe_minus>": ["", "-"],
    "<non_zero_digit>": [str(num) for num in range(1, 10)],  # Exclude 0 from starting digits
    "<digit>": list(string.digits),
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<maybe_frac>": ["", ".<digits>"],
}


if __name__ == "__main__":
    default_param = {
        "log": True,
        "max_iterations": 5,
        "grammar": divide_by_zero_grammar,
        "initial_inputs": divide_by_zero_inputs,
        "oracle": divide_by_zero_oracle,
    }

    avicenna = Avicenna(
        **default_param,
        # pattern_learner=AvicennaPatternLearner
    )

    diagnosis = avicenna.explain()
    print("Final Diagnosis:")
    print(ISLaUnparser(diagnosis[0]).unparse())

    equivalent_representations = avicenna.get_equivalent_best_formulas()

    if equivalent_representations:
        print("\nEquivalent Representations:")
        for diagnosis in equivalent_representations:
            print(ISLaUnparser(diagnosis[0]).unparse())

    print("All Learned Formulas (that meet min criteria)", end="\n\n")
    cand = avicenna.get_learned_formulas()
    for can in cand:
        print(f"Avicenna calculated a precision of {can[1] * 100:.2f}% and a recall of {can[2] * 100:.2f}%")
        print(ISLaUnparser(can[0]).unparse(), end="\n\n")
        print(len(can[0]))

