import unittest
from typing import Set
from fuzzingbook.Parser import EarleyParser
from isla.language import ISLaUnparser, DerivationTree

from avicenna_formalizations.calculator import grammar, arith_eval
from avicenna_formalizations import get_pattern_file_path
from avicenna.islearn import AvicennaISlearn
from avicenna.input import Input
from avicenna.oracle import OracleResult

def oracle(inp: DerivationTree) -> bool:
    try:
        arith_eval(str(inp))
        return False
    except ValueError:
        return True

class TestAvicennaIslearn(unittest.TestCase):
    @unittest.skip
    def test_islearn(self):
        test_inputs: Set[Input] = set()
        test_inputs.add(
            Input(
                DerivationTree.from_parse_tree(
                    next(EarleyParser(grammar).parse('sqrt(-900)')
                         )
                ), OracleResult.BUG
            )
        )


        print(get_pattern_file_path())
        islearn = AvicennaISlearn(
            grammar=grammar,
            prop=oracle,
            pattern_file=str(get_pattern_file_path()),
        )

        result = islearn.learn_failure_invariants(
            test_inputs=test_inputs
        )
        failure_constraints = list(
            map(lambda p: (p[1], ISLaUnparser(p[0]).unparse()), result.items())
        )

        result = islearn.learn_failure_invariants(
            test_inputs=test_inputs
        )
        failure_constraints = list(
            map(lambda p: (p[1], ISLaUnparser(p[0]).unparse()), result.items())
        )


if __name__ == '__main__':
    unittest.main()
