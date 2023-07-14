import unittest
from pathlib import Path

from fuzzingbook.Grammars import is_valid_grammar
from fuzzingbook.Parser import EarleyParser
from isla.language import DerivationTree
from isla.language import ISLaUnparser
from islearn.learner import InvariantLearner

from avicenna.feature_extractor import Extractor
from avicenna.feature_collector import Collector
from avicenna.generator import Generator
from avicenna_formalizations.calculator import prop as prop_alhazen, grammar as CALCULATOR_GRAMMAR
from avicenna_formalizations.heartbeat import grammar as HEARTBLEED, prop

MAX_POSITIVE_SAMPLES = 60
MAX_NEGATIVE_SAMPLES = 60


class RelevantInputExtraction(unittest.TestCase):
    @unittest.skip
    def test_feature_extraction(self):

        assert is_valid_grammar(HEARTBLEED)

        g = Generator(MAX_POSITIVE_SAMPLES, MAX_NEGATIVE_SAMPLES, HEARTBLEED, prop)
        positive_sample = "8 pasbd xyasd"
        positive_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(HEARTBLEED).parse(inp)))
            for inp in [positive_sample]
        ]

        val, neg_val = g.generate_mutation(positive_trees)

        assert len(val) == MAX_POSITIVE_SAMPLES
        assert len(neg_val) == MAX_NEGATIVE_SAMPLES

        sample_list = val + neg_val
        collector = Collector(HEARTBLEED)
        feature_table = collector.collect_features(sample_list)
        oracle_table = [prop(inp) for inp in sample_list]
        assert all(oracle_table[0:MAX_POSITIVE_SAMPLES]) is True
        assert all(oracle_table[MAX_POSITIVE_SAMPLES:]) is False

        combined_data = feature_table.copy()
        combined_data["oracle"] = oracle_table

        extractor = Extractor(combined_data, HEARTBLEED, max_feature_num=2)
        ex = extractor.extract_non_terminals()

        final_results = extractor.get_clean_most_important_features()
        f_s = set()
        for feature in extractor.get_clean_most_important_features():
            f_s.add(feature.rule)

        self.assertTrue(
            all([elem in {"<length>", "<payload>"} for elem in f_s]),
            f"{f_s} is not the expected output.",
        )

    @unittest.skip
    def test_feature_extraction_calculator(self):
        assert is_valid_grammar(CALCULATOR_GRAMMAR)

        g = Generator(
            MAX_POSITIVE_SAMPLES, MAX_NEGATIVE_SAMPLES, CALCULATOR_GRAMMAR, prop_alhazen
        )
        initial_inputs_alhazen = ["sqrt(-900)", "sqrt(-123)"]
        positive_trees = [
            DerivationTree.from_parse_tree(
                next(EarleyParser(CALCULATOR_GRAMMAR).parse(inp))
            )
            for inp in initial_inputs_alhazen
        ]

        val, neg_val = g.generate_mutation(positive_trees)

        assert len(val) == MAX_POSITIVE_SAMPLES
        assert len(neg_val) == MAX_NEGATIVE_SAMPLES

        sample_list = val + neg_val
        collector = Collector(CALCULATOR_GRAMMAR)
        feature_table = collector.collect_features(sample_list)
        oracle_table = [prop_alhazen(inp) for inp in sample_list]
        assert all(oracle_table[0:MAX_POSITIVE_SAMPLES]) is True
        assert all(oracle_table[MAX_POSITIVE_SAMPLES:]) is False

        combined_data = feature_table.copy()
        combined_data["oracle"] = oracle_table

        extractor = Extractor(combined_data, CALCULATOR_GRAMMAR, max_feature_num=2)
        ex = extractor.extract_non_terminals()

        final_results = extractor.get_clean_most_important_features()
        f_s = set()
        for feature in extractor.get_clean_most_important_features():
            f_s.add(feature.rule)

        self.assertTrue(
            all(
                [
                    elem in {"<number>", "<function>"}
                    or elem in {"<maybe_minus>", "<function>"}
                    for elem in f_s
                ]
            ),
            f"{f_s} is not the expected output.",
        )

    @unittest.skip
    def test_islearn_with_exclusion_set(self):
        expected_result = """exists <payload> container1 in start:\n  exists <length> length_field in start:\n    (< (str.len container1) (str.to.int length_field))"""

        # Without <length> and <payload>
        exclusion_set = {
            "<string>",
            "<maybe_digits>",
            "<start>",
            "<onenine>",
            "<digits>",
            "<char>",
            "<padding>",
            "<digit>",
        }

        g = Generator(MAX_POSITIVE_SAMPLES, MAX_NEGATIVE_SAMPLES, HEARTBLEED, prop)
        positive_sample = "8 pasbd xyasd"
        positive_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(HEARTBLEED).parse(inp)))
            for inp in [positive_sample]
        ]
        val, neg_val = g.generate_mutation(positive_trees)

        result = InvariantLearner(
            grammar=HEARTBLEED,
            prop=prop,
            pattern_file=str(Path("../src/avicenna_formalizations/patterns.toml").resolve()),
            positive_examples=val,
            negative_examples=neg_val,
            exclude_nonterminals=exclusion_set,
            target_number_positive_samples=50,
            target_number_positive_samples_for_learning=50,
            target_number_negative_samples=50,
            min_recall=1.0,
            min_specificity=1.0,
            filter_inputs_for_learning_by_kpaths=False,
        ).learn_invariants()

        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse() + "\n", result.items())))

        self.assertTrue(
            len(result) == 1,
            f"too many constraints learned. Expected 1, got {len(result)}.",
        )
        self.assertIn(
            expected_result.strip(),
            map(lambda f: ISLaUnparser(f).unparse(), result.keys()),
        )


if __name__ == "__main__":
    unittest.main()
