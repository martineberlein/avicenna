import unittest
from flaky import flaky
from typing import Set

from isla.fuzzer import GrammarFuzzer

from avicenna_formalizations.calculator import (
    grammar as grammar_calculator,
    oracle as oracle_calculator,
)
from avicenna.input import Input
from avicenna.oracle import OracleResult
from avicenna.feature_collector import GrammarFeatureCollector
from avicenna import feature_extractor
from avicenna.features import (
    ExistenceFeature,
    NumericFeature,
    LengthFeature,
    DerivationFeature,
)
from avicenna.monads import Exceptional


class TestRelevantFeatureLearner(unittest.TestCase):
    def setUp(self) -> None:
        inputs = [
            ("sqrt(-901)", OracleResult.BUG),
            ("sqrt(-1)", OracleResult.BUG),
            ("sqrt(10)", OracleResult.NO_BUG),
            ("cos(1)", OracleResult.NO_BUG),
            ("sin(99)", OracleResult.NO_BUG),
            ("tan(-20)", OracleResult.NO_BUG),
        ]
        collector = GrammarFeatureCollector(grammar_calculator)

        self.test_inputs = (
            Exceptional.of(lambda: inputs)
            .map(
                lambda x: {
                    Input.from_str(grammar_calculator, inp_, oracle=orc_)
                    for inp_, orc_ in x
                }
            )
            .map(
                lambda x: {
                    inp_.update_features(collector.collect_features(inp_)) for inp_ in x
                }
            )
            .reraise()
            .get()
        )

    def test_relevant_feature_learner(self):
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            grammar_calculator
        )
        (
            relevant_features,
            correlating_features,
            excluded_features,
        ) = feature_learner.learn(self.test_inputs)
        self.assertNotEqual(len(relevant_features), 0)
        self.assertNotEqual(len(correlating_features), 0)
        self.assertNotEqual(len(excluded_features), 0)

        expected_features = {
            NumericFeature("<number>"),
            DerivationFeature("<function>", "sqrt"),
            DerivationFeature("<maybe_minus>", "-"),
        }

        # Check that all expected features are identified as either relevant or correlating.
        self.assertTrue(
            all(
                feature in relevant_features.union(correlating_features)
                for feature in expected_features
            )
        )

    def test_empty_input_set(self):
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            grammar_calculator
        )
        test_inputs: Set[Input] = set()
        with self.assertRaises(ValueError):
            _ = feature_learner.learn(test_inputs)

    def test_relevant_feature_exception_handling(self):
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            grammar_calculator
        )

        excluded_non_terminal_strings = (
            Exceptional.of(lambda: self.test_inputs)
            .map(lambda inputs: feature_learner.learn(inputs))
            .map(lambda x: x[0].union(x[1]))
            .map(lambda x: {feature.non_terminal for feature in x})
            .map(lambda x: set(grammar_calculator.keys()).difference(x))
            .reraise()
            .get()
        )

        relevant_features = {
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
                for feature in relevant_features
            )
        )

    def test_learner_identifies_expected_features_with_large_data(self):
        fuzzer = GrammarFuzzer(grammar_calculator)
        collector = GrammarFeatureCollector(grammar_calculator)
        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            grammar_calculator
        )

        excluded_non_terminal_strings = (
            Exceptional.of(lambda: None)
            .map(lambda _: {fuzzer.fuzz_tree() for _ in range(100)})
            .map(
                lambda fuzzing_inputs: {
                    Input(tree=inp_, oracle=oracle_calculator(str(inp_)))
                    for inp_ in fuzzing_inputs
                }
            )
            .map(
                lambda parsed_inputs: {
                    inp_.update_features(collector.collect_features(inp_))
                    for inp_ in parsed_inputs
                }
            )
            .map(lambda parsed_inputs: feature_learner.learn(parsed_inputs))
            .map(lambda learning_data: learning_data[0].union(learning_data[1]))
            .map(
                lambda relevant_features_: {
                    feature.non_terminal for feature in relevant_features_
                }
            )
            .map(
                lambda relevant_non_terminals: set(
                    grammar_calculator.keys()
                ).difference(relevant_non_terminals)
            )
            .reraise()
            .get()
        )

        relevant_features = {
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
                for feature in relevant_features
            )
        )

    @flaky(max_runs=3, min_passes=2)
    def test_relevant_feature_learner_middle(self):
        from avicenna_formalizations.middle import grammar, oracle
        features = [
            ExistenceFeature,
            NumericFeature,
            DerivationFeature,
        ]
        fuzzer = GrammarFuzzer(grammar)
        collector = GrammarFeatureCollector(grammar, feature_types=features)
        feature_learner = feature_extractor.SHAPRelevanceLearner(
            grammar,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
            feature_types=features,
        )

        excluded_non_terminal_strings = (
            Exceptional.of(lambda: None)
            .map(lambda _: {fuzzer.fuzz_tree() for _ in range(100)})
            .map(
                lambda fuzzing_inputs: {
                    Input(tree=inp_, oracle=oracle(str(inp_)))
                    for inp_ in fuzzing_inputs
                }
            )
            .map(
                lambda parsed_inputs: {
                    inp_.update_features(collector.collect_features(inp_))
                    for inp_ in parsed_inputs
                }
            )
            .map(lambda parsed_inputs: feature_learner.learn(parsed_inputs))
            .map(lambda learning_data: learning_data[0].union(learning_data[1]))
            .map(
                lambda relevant_features_: {
                    feature.non_terminal for feature in relevant_features_
                }
            )
            .map(
                lambda relevant_non_terminals: set(
                    grammar_calculator.keys()
                ).difference(relevant_non_terminals)
            )
            .reraise()
            .get()
        )

        relevant_features = {
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
                for feature in relevant_features
            )
        )


    def test_learner_heartbleed(self):
        from avicenna_formalizations.heartbeat import grammar, oracle

        fuzzer = GrammarFuzzer(grammar)
        collector = GrammarFeatureCollector(grammar)
        feature_learner = feature_extractor.SHAPRelevanceLearner(
            grammar,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
            top_n=2,
        )

        excluded_non_terminal_strings = (
            Exceptional.of(lambda: None)
            .map(lambda _: {fuzzer.fuzz_tree() for _ in range(100)})
            .map(
                lambda fuzzing_inputs: {
                    Input(tree=inp_, oracle=oracle(str(inp_)))
                    for inp_ in fuzzing_inputs
                }
            )
            .map(
                lambda parsed_inputs: {
                    inp_.update_features(collector.collect_features(inp_))
                    for inp_ in parsed_inputs
                }
            )
            .map(lambda parsed_inputs: feature_learner.learn(parsed_inputs))
            .map(lambda learning_data: learning_data[0].union(learning_data[1]))
            .map(
                lambda relevant_features_: {
                    feature.non_terminal for feature in relevant_features_
                }
            )
            .map(
                lambda relevant_non_terminals: set(
                    grammar_calculator.keys()
                ).difference(relevant_non_terminals)
            )
            .reraise()
            .get()
        )

        relevant_features = {
            NumericFeature("<payload-length>"),
            LengthFeature("<payload>")
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
                for feature in relevant_features
            )
        )

    @unittest.skip
    def test_learner_xml(self):
        from isla.derivation_tree import DerivationTree
        from isla_formalizations import xml_lang

        def xml_oracle(tree: DerivationTree) -> OracleResult:
            if xml_lang.validate_xml(tree) is False:
                return OracleResult.BUG
            else:
                return OracleResult.NO_BUG

        fuzzer = GrammarFuzzer(xml_lang.XML_GRAMMAR)
        test_inputs = set()
        for _ in range(100):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(tree=inp, oracle=xml_oracle(inp)))

        test_inputs.update(
            set(
                [
                    Input.from_str(xml_lang.XML_GRAMMAR, inp, OracleResult.NO_BUG)
                    for inp in ["<a>as</b>", "<c>Text</c>"]
                ]
            )
        )

        collector = GrammarFeatureCollector(xml_lang.XML_GRAMMAR)
        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        feature_learner = feature_extractor.SHAPRelevanceLearner(
            xml_lang.XML_GRAMMAR,
            classifier_type=feature_extractor.RandomForestRelevanceLearner,
        )
        relevant_features, corr, ex = feature_learner.learn(test_inputs)
        print(relevant_features)
        print(corr)
        print(ex)


if __name__ == "__main__":
    unittest.main()
