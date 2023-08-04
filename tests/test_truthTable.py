import unittest
from typing import cast

from isla import language
from isla import isla_shortcuts as sc
from isla.z3_helpers import z3_eq
import z3


from avicenna.result_table import TruthTable, TruthTableRow


class MyTestCase(unittest.TestCase):
    def test_something(self):
        start = language.Constant("$start", "<start>")
        var1 = language.BoundVariable("$var", "<var>")

        formula = sc.forall(
            var1,
            start,
            sc.smt_for(cast(z3.BoolRef, z3_eq(var1.to_smt(), z3.StringVal("x"))), var1),
        )

        r = TruthTable([TruthTableRow(formula), TruthTableRow(formula)])
        p = TruthTable()
        p.append(TruthTableRow(formula))


if __name__ == "__main__":
    unittest.main()
