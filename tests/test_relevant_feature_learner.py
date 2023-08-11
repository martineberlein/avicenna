import unittest

from isla.fuzzer import GrammarFuzzer

from avicenna_formalizations.calculator import (
    grammar as grammar_calculator,
    oracle as oracle_calculator,
)
from avicenna.input import Input
from avicenna.oracle import OracleResult
from avicenna.feature_collector import GrammarFeatureCollector
from avicenna import feature_extractor
from avicenna.features import ExistenceFeature, NumericFeature, LengthFeature, DerivationFeature


class TestRelevantFeatureLearner(unittest.TestCase):
    def test_relevant_feature_learner(self):
        inputs = [
            ("sqrt(-901)", OracleResult.BUG),
            ("sqrt(-1)", OracleResult.BUG),
            ("sqrt(10)", OracleResult.NO_BUG),
            ("cos(1)", OracleResult.NO_BUG),
            ("sin(99)", OracleResult.NO_BUG),
            ("tan(-20)", OracleResult.NO_BUG),
        ]
        test_inputs = set(
            [Input.from_str(grammar_calculator, inp, oracle) for inp, oracle in inputs]
        )
        collector = GrammarFeatureCollector(grammar_calculator)
        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        feature_learner = feature_extractor.DecisionTreeRelevanceLearner(
            grammar_calculator
        )
        relevant_features, corr, ex = feature_learner.learn(test_inputs)
        print(relevant_features)
        print(corr)
        print(ex)

    def test_relevant_feature_learner_more_data(self):
        fuzzer = GrammarFuzzer(grammar_calculator)
        test_inputs = set()
        for _ in range(200):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(tree=inp, oracle=oracle_calculator(str(inp))))

        collector = GrammarFeatureCollector(grammar_calculator)
        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        feature_learner = feature_extractor.SHAPRelevanceLearner(
            grammar_calculator,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
        )
        relevant_features, corr, ex = feature_learner.learn(test_inputs)
        print(relevant_features)
        print(corr)
        print(ex)

    def test_relevant_feature_learner_middle(self):
        from avicenna_formalizations.middle import grammar, oracle
        features = [
            ExistenceFeature,
            NumericFeature,
            DerivationFeature,
            #LengthFeature
        ]

        fuzzer = GrammarFuzzer(grammar)
        test_inputs = set()
        for _ in range(100):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(tree=inp, oracle=oracle(str(inp))))

        p = [inp for inp in test_inputs if inp.oracle == OracleResult.BUG]
        n = [inp for inp in test_inputs if inp.oracle == OracleResult.NO_BUG]
        test_inputs = set(p + n)

        collector = GrammarFeatureCollector(grammar, feature_types=features)
        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        feature_learner = feature_extractor.SHAPRelevanceLearner(
            grammar,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
            feature_types=features
        )
        relevant_features, corr, ex = feature_learner.learn(test_inputs)
        print(relevant_features)
        print(corr)
        print(ex)
        print(len([inp for inp in test_inputs if inp.oracle == OracleResult.BUG]))
        print(len([inp for inp in test_inputs if inp.oracle == OracleResult.NO_BUG]))

    def test_learner_heartbleed(self):
        from avicenna_formalizations.heartbeat import grammar, oracle
        features = [
            ExistenceFeature,
            NumericFeature,
            DerivationFeature,
            LengthFeature
        ]

        fuzzer = GrammarFuzzer(grammar)
        test_inputs = set()
        for _ in range(100):
            inp = fuzzer.fuzz_tree()
            test_inputs.add(Input(tree=inp, oracle=oracle(str(inp))))

        collector = GrammarFeatureCollector(grammar, features)
        for inp in test_inputs:
            inp.features = collector.collect_features(inp)

        feature_learner = feature_extractor.SHAPRelevanceLearner(
            grammar,
            classifier_type=feature_extractor.GradientBoostingTreeRelevanceLearner,
            feature_types=features,
            top_n=2,
            show_beeswarm_plot=True,
        )

        relevant_features, corr, ex = feature_learner.learn(test_inputs)
        print(relevant_features)
        print(corr)
        print(ex)
        print(len([inp for inp in test_inputs if inp.oracle == OracleResult.BUG]))
        print(len([inp for inp in test_inputs if inp.oracle == OracleResult.NO_BUG]))

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
