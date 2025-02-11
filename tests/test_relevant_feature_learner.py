import unittest
from typing import Set

from isla.fuzzer import GrammarFuzzer
from debugging_framework.input.oracle import OracleResult

from avicenna.data import Input
from avicenna.features.feature_collector import GrammarFeatureCollector
import avicenna.learning.reducer as feature_extractor
from avicenna.features.features import (
    ExistenceFeature,
    NumericFeature,
    LengthFeature,
    DerivationFeature
)

from resources.subjects import get_calculator_subject, get_middle_subject, get_heartbleed_subject


class TestRelevantFeatureLearner(unittest.TestCase):
    def setUp(self) -> None:
        inputs = [
            ("sqrt(-901)", OracleResult.FAILING),
            ("sqrt(-1)", OracleResult.FAILING),
            ("sqrt(-6)", OracleResult.FAILING),
            ("sqrt(10)", OracleResult.PASSING),
            ("cos(1)", OracleResult.PASSING),
            ("sin(99)", OracleResult.PASSING),
            ("sin(4)", OracleResult.PASSING),
            ("tan(-20)", OracleResult.PASSING),
        ]

        self.calculator = get_calculator_subject()
        self.collector = GrammarFeatureCollector(self.calculator.grammar)

        parsed_inputs = {
            Input.from_str(self.calculator.grammar, inp_, oracle=orc_)
            for inp_, orc_ in inputs
        }
        self.test_inputs = {
            inp_.update_features(self.collector.collect_features(inp_)) for inp_ in parsed_inputs
        }

    def test_relevant_feature_learner(self):
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            self.calculator.grammar, prune_parent_correlation=False
        )
        relevant_features = feature_learner.learn(self.test_inputs)
        self.assertNotEqual(len(relevant_features), 0)

        expected_relevant_features = {
            NumericFeature("<number>"),
            DerivationFeature("<function>", "sqrt"),
            DerivationFeature("<maybe_minus>", "-"),
        }

        # Check that all expected features are identified as either relevant or correlating.
        self.assertTrue(
            all(
                feature in relevant_features
                for feature in expected_relevant_features
            )
        )

    def test_empty_input_set(self):
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            self.calculator.grammar
        )
        test_inputs: Set[Input] = set()
        with self.assertRaises(ValueError):
            _ = feature_learner.learn(test_inputs)

    def test_relevant_feature_exclusion(self):
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            self.calculator.grammar, prune_parent_correlation=True
        )

        relevant_features = feature_learner.learn(self.test_inputs)
        relevant_features_non_terminals = {feature.non_terminal for feature in relevant_features}
        excluded_non_terminal_strings = set(self.calculator.grammar.keys()).difference(relevant_features_non_terminals)

        expected_relevant_features = {
            NumericFeature("<number>"),
            DerivationFeature("<function>", "sqrt"),
            DerivationFeature("<maybe_minus>", "-"),
        }

        self.assertNotEqual(
            len(excluded_non_terminal_strings),
            0,
            "Expected at least one non-terminals to be excluded",
        )
        # Check that all relevant features are not in the set of excluded non_terminals
        # Check that all relevant features are not in the set of excluded non_terminals
        self.assertTrue(
            all(
                feature.non_terminal not in excluded_non_terminal_strings
                for feature in expected_relevant_features
            ), f"Expected relevant features: {expected_relevant_features}, but got: {relevant_features}"
        )

    def test_learner_identifies_expected_features_with_large_data(self):
        fuzzer = GrammarFuzzer(self.calculator.grammar)
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            self.calculator.grammar, prune_parent_correlation=False
        )

        test_inputs = {fuzzer.fuzz_tree() for _ in range(100)}
        parsed_inputs = {Input(tree=inp, oracle=self.calculator.oracle(str(inp))[0]) for inp in test_inputs}
        feature_inputs = {
            inp_.update_features(self.collector.collect_features(inp_)) for inp_ in parsed_inputs
        }

        relevant_features = feature_learner.learn(feature_inputs)
        relevant_features_non_terminals = {feature.non_terminal for feature in relevant_features}
        excluded_non_terminal_strings = set(self.calculator.grammar.keys()).difference(relevant_features_non_terminals)

        expected_relevant_features = {
            NumericFeature("<number>"),
            DerivationFeature("<function>", "sqrt"),
            DerivationFeature("<maybe_minus>", "-"),
        }

        self.assertNotEqual(
            len(excluded_non_terminal_strings),
            0,
            "Expected at least one non-terminals to be excluded",
        )
        # Check that all relevant features are not in the set of excluded non_terminals
        self.assertTrue(
            all(
                feature.non_terminal not in excluded_non_terminal_strings
                for feature in expected_relevant_features
            ), f"Expected relevant features: {expected_relevant_features}, but got: {relevant_features}"
        )

    def test_relevant_feature_learner_middle(self):
        middle = get_middle_subject()

        features = [
            ExistenceFeature,
            NumericFeature,
            DerivationFeature,
        ]
        fuzzer = GrammarFuzzer(middle.grammar)
        collector = GrammarFeatureCollector(middle.grammar, feature_types=features)
        feature_learner = feature_extractor.SHAPRelevanceLearner(
            middle.grammar,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
            feature_types=features,
            top_n_relevant_features=4,
        )

        test_inputs = {fuzzer.fuzz_tree() for _ in range(200)}
        test_inputs = {
                    Input(tree=inp_, oracle=middle.oracle(str(inp_))[0])
                    for inp_ in test_inputs
                }

        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        relevant_features = feature_learner.learn(test_inputs)
        relevant_feature_non_terminals = {feature.non_terminal for feature in relevant_features}
        excluded_non_terminal_strings = set(
            middle.grammar.keys()
        ).difference(relevant_feature_non_terminals)

        expected_relevant_features = {
            NumericFeature("<x>"),
            NumericFeature("<y>"),
            NumericFeature("<z>"),
        }

        self.assertNotEqual(
            len(excluded_non_terminal_strings),
            0,
            "Expected at least one non-terminals to be excluded",
        )
        # Check that all relevant features are not in the set of excluded non_terminals
        self.assertTrue(
            all(
                feature.non_terminal not in excluded_non_terminal_strings
                for feature in expected_relevant_features
            ), f"Expected relevant features: {expected_relevant_features}, but got: {relevant_features}"
        )

    def test_learner_heartbleed(self):
        heartbleed = get_heartbleed_subject()

        fuzzer = GrammarFuzzer(heartbleed.grammar)
        collector = GrammarFeatureCollector(heartbleed.grammar)
        feature_learner = feature_extractor.SHAPRelevanceLearner(
            heartbleed.grammar,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
            top_n_relevant_features=2,
        )

        test_inputs = {fuzzer.fuzz_tree() for _ in range(100)}
        test_inputs = {
                    Input(tree=inp_, oracle=heartbleed.oracle(str(inp_))[0])
                    for inp_ in test_inputs
                }

        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        relevant_features = feature_learner.learn(test_inputs)
        relevant_feature_non_terminals = {feature.non_terminal for feature in relevant_features}
        excluded_non_terminal_strings = set(
            heartbleed.grammar.keys()
        ).difference(relevant_feature_non_terminals)

        expected_relevant_features = {
            NumericFeature("<payload-length>"),
            LengthFeature("<payload>"),
        }

        self.assertNotEqual(
            len(excluded_non_terminal_strings),
            0,
            "Expected at least one non-terminals to be excluded",
        )
        # Check that all relevant features are not in the set of excluded non_terminals
        self.assertTrue(
            all(
                feature.non_terminal not in excluded_non_terminal_strings
                for feature in expected_relevant_features
            ), f"Expected relevant features: {expected_relevant_features}, but got: {relevant_features}"
        )

    def test_shap_feature_learner_with_special_characters(self):
        grammar_with_json_chars = {
            "<start>": ["<arg>"],
            "<arg>": ["<digit>", '"<digit>"'],
            "<digit>": ["1", "2", "3"],
        }

        inputs = [
            ("1", OracleResult.PASSING),
            ("2", OracleResult.PASSING),
            ('"3"', OracleResult.FAILING),
        ]

        collector = GrammarFeatureCollector(grammar_with_json_chars)
        feature_learner = feature_extractor.SHAPRelevanceLearner(
            grammar_with_json_chars,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
        )

        parsed_inputs = {
            Input.from_str(grammar_with_json_chars, inp_, oracle=orc_)
            for inp_, orc_ in inputs
        }
        feature_inputs = {
            inp_.update_features(collector.collect_features(inp_))
            for inp_ in parsed_inputs
        }
        relevant_features = feature_learner.learn(feature_inputs)

        self.assertNotEqual(
            len(relevant_features),
            0,
            "Expected at least one non-terminals to be relevant.",
        )

    def test_learner_xml(self):
        from isla.derivation_tree import DerivationTree
        from isla_formalizations import xml_lang

        def xml_oracle(tree: DerivationTree) -> OracleResult:
            if xml_lang.validate_xml(tree) is False:
                return OracleResult.FAILING
            else:
                return OracleResult.PASSING

        fuzzer = GrammarFuzzer(xml_lang.XML_GRAMMAR)
        test_inputs = set()
        for _ in range(100):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(tree=inp, oracle=xml_oracle(inp)))

        test_inputs.update(
            set(
                [
                    Input.from_str(xml_lang.XML_GRAMMAR, inp, OracleResult.PASSING)
                    for inp in ["<a>as</b>", "<c>Text</c>"]
                ]
            )
        )

        collector = GrammarFeatureCollector(xml_lang.XML_GRAMMAR)
        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        feature_learner = feature_extractor.RandomForestRelevanceLearner(
            xml_lang.XML_GRAMMAR,
        )
        relevant_features = feature_learner.learn(test_inputs)
        print(relevant_features)


if __name__ == "__main__":
    unittest.main()
