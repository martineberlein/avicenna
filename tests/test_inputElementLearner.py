import unittest
import logging

from fuzzingbook.Parser import EarleyParser
from isla.language import DerivationTree
from isla_formalizations import xml_lang

from avicenna.learner import InputElementLearner
from avicenna.input import Input

from avicenna_formalizations.calculator import (
    prop as prop_alhazen,
    grammar as grammar_calculator,
)
from avicenna_formalizations.heartbeat import (
    prop_ as prop_heartbeat,
    grammar as grammar_heartbeat,
)


class TestInputElementLearner(unittest.TestCase):
    @unittest.skip
    def test_learner_calculator(self):
        inputs = ["sqrt(-900)"]
        test_inputs = set()
        for inp in inputs:
            test_inputs.add(
                Input(
                    DerivationTree.from_parse_tree(
                        next(EarleyParser(grammar_calculator).parse(inp))
                    )
                )
            )

        result = InputElementLearner(
            grammar=grammar_calculator,
            oracle=prop_alhazen,
            max_relevant_features=2,
        ).learn(test_inputs=test_inputs)
        non_terminals = [elem[0] for elem in result]

        self.assertTrue(
            all(
                [
                    elem in {"<number>", "<function>"}
                    or elem in {"<maybe_minus>", "<function>"}
                    for elem in non_terminals
                ]
            ),
            f"{non_terminals} is not the expected output.",
        )

    @unittest.skip
    def test_learner_heartbeat(self):
        inputs = ["8 pasbd xyasd", "2 xy kjsdfh"]
        test_inputs = set()
        for inp in inputs:
            test_inputs.add(
                Input(
                    DerivationTree.from_parse_tree(
                        next(EarleyParser(grammar_heartbeat).parse(inp))
                    )
                )
            )

        result = InputElementLearner(
            grammar_heartbeat,
            prop_heartbeat,
            input_samples=test_inputs,
            max_relevant_features=2,
        ).learn()
        non_terminals = [elem[0] for elem in result]

        self.assertTrue(
            all([elem in {"<length>", "<payload>"} for elem in non_terminals]),
            f"{non_terminals} is not " f"the expected output.",
        )

    @unittest.skip
    def test_learner_xml(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")

        def prop(tree: DerivationTree) -> bool:
            return xml_lang.validate_xml(tree) is False

        result = InputElementLearner(
            xml_lang.XML_GRAMMAR,
            prop,
            input_samples=["<a>as</b>", "<c>Text</c>"],
            max_relevant_features=4,
        ).learn()
        non_terminals = [elem[0] for elem in result]
        print(result)
        print(non_terminals)


if __name__ == "__main__":
    unittest.main()
