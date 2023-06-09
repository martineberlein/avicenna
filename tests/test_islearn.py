import unittest
from pathlib import Path
from fuzzingbook.Parser import EarleyParser
from isla.language import ISLaUnparser, DerivationTree

from avicenna_formalizations.calculator import grammar, arith_eval
from avicenna_formalizations import get_pattern_file_path

from avicenna.islearn import AvicennaISlearn

def oracle(inp: DerivationTree) -> bool:
    try:
        arith_eval(str(inp))
        return False
    except ValueError:
        return True

class TestAvicennaIslearn(unittest.TestCase):
    def test_islearn(self):
        positive_samples = list()
        positive_samples.append(DerivationTree.from_parse_tree(
            next(EarleyParser(grammar).parse('sqrt(-900)'))
        ))

        islearn = AvicennaISlearn(
            grammar=grammar,
            prop=oracle,
            pattern_file=str(get_pattern_file_path()),
        )

        result = islearn.learn_failure_invariants(
            positive_examples=positive_samples
        )
        failure_constraints = list(
            map(lambda p: (p[1], ISLaUnparser(p[0]).unparse()), result.items())
        )
        #for i in failure_constraints:
        #    print(i)

        result = islearn.learn_failure_invariants(
            positive_examples=positive_samples
        )
        failure_constraints = list(
            map(lambda p: (p[1], ISLaUnparser(p[0]).unparse()), result.items())
        )


if __name__ == '__main__':
    unittest.main()
