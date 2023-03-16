import unittest
import logging

from isla.language import DerivationTree
from isla_formalizations import xml_lang

from avicenna.learner import InputElementLearner

from avicenna_formalizations.calculator import prop as prop_alhazen, CALCULATOR_GRAMMAR
from avicenna_formalizations.heartbeat import prop as prop_heartbeat, HEARTBLEED


class TestInputElementLearner(unittest.TestCase):
    def test_learner_calculator(self):

        result = InputElementLearner(
            CALCULATOR_GRAMMAR,
            prop_alhazen,
            input_samples=["sqrt(-900)"],
            max_relevant_features=2,
        ).learn()
        non_terminals = [elem[0] for elem in result]
        print(result)

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

    def test_learner_heartbeat(self):

        result = InputElementLearner(
            HEARTBLEED,
            prop_heartbeat,
            input_samples=["8 pasbd xyasd"],
            max_relevant_features=2,
        ).learn()
        non_terminals = [elem[0] for elem in result]
        print(result)

        self.assertTrue(
            all([elem in {"<length>", "<payload>"} for elem in non_terminals]),
            f"{non_terminals} is not " f"the expected output.",
        )

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
