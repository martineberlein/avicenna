import unittest

from avicenna_formalizations.calculator import (
    grammar as grammar_calculator,
    oracle as oracle_calculator,
)

from avicenna.input import Input
from avicenna.oracle import OracleResult
from avicenna.feature_collector import GrammarFeatureCollector
from avicenna import feature_extractor

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
        from isla.fuzzer import GrammarFuzzer

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


if __name__ == "__main__":
    unittest.main()
