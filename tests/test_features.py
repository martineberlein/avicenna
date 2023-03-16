import string
import unittest

import numpy
import pandas as pd
from fuzzingbook.Grammars import Grammar, is_valid_grammar, srange
from fuzzingbook.Parser import EarleyParser
from isla.language import DerivationTree

from avicenna.feature_collector import Collector
from avicenna.features import (
    EXISTENCE_FEATURE,
    NUMERIC_INTERPRETATION_FEATURE,
    LENGTH_FEATURE,
    IS_DIGIT_FEATURE,
)

STANDARD_FEATURES = {EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE, LENGTH_FEATURE}

grammar: Grammar = {
    "<start>": ["<string>"],
    "<string>": ["<A>", "<B>", "!ab!"],
    "<A>": srange(string.ascii_lowercase),
    "<B>": srange(string.digits),
}
assert is_valid_grammar(grammar)

grammar_rec: Grammar = {
    "<start>": ["<string>"],
    "<string>": ["<A>", "<B>", "!ab!"],
    "<A>": ["<chars><A>", ""],
    "<chars>": srange(string.ascii_lowercase),
    "<B>": ["<digit><B>", ""],
    "<digit>": srange(string.digits),
}
assert is_valid_grammar(grammar_rec)

grammar_with_maybe_minus: Grammar = {
    "<start>": ["<string>"],
    "<string>": ["<A>" "<B>"],
    "<A>": ["", "-"],
    "<B>": srange(string.digits),
}
assert is_valid_grammar(grammar_with_maybe_minus)


class FeatureExtraction(unittest.TestCase):
    def test_extract_features(self):
        expected_feature_list = {
            "num(<B>)",
            "len(<start>)",
            "len(<string>)",
            "len(<A>)",
            "len(<B>)",
            "exists(<start>)",
            "exists(<start>@0)",
            "exists(<string>)",
            "exists(<string>@0)",
            "exists(<string>@1)",
            "exists(<string>@2)",
            "exists(<A>)",
            "exists(<A>@0)",
            "exists(<A>@1)",
            "exists(<A>@2)",
            "exists(<A>@3)",
            "exists(<A>@4)",
            "exists(<A>@5)",
            "exists(<A>@6)",
            "exists(<A>@7)",
            "exists(<A>@8)",
            "exists(<A>@9)",
            "exists(<A>@10)",
            "exists(<A>@11)",
            "exists(<A>@12)",
            "exists(<A>@13)",
            "exists(<A>@14)",
            "exists(<A>@15)",
            "exists(<A>@16)",
            "exists(<A>@17)",
            "exists(<A>@18)",
            "exists(<A>@19)",
            "exists(<A>@20)",
            "exists(<A>@21)",
            "exists(<A>@22)",
            "exists(<A>@23)",
            "exists(<A>@24)",
            "exists(<A>@25)",
            "exists(<B>)",
            "exists(<B>@0)",
            "exists(<B>@1)",
            "exists(<B>@2)",
            "exists(<B>@3)",
            "exists(<B>@4)",
            "exists(<B>@5)",
            "exists(<B>@6)",
            "exists(<B>@7)",
            "exists(<B>@8)",
            "exists(<B>@9)",
        }

        expected_feature_list_strings = {
            "exists(<A> == q)",
            "exists(<A> == y)",
            "exists(<A> == l)",
            "exists(<A> == k)",
            "exists(<B> == 9)",
            "len(<start>)",
            "exists(<A>)",
            "exists(<start> == <string>)",
            "exists(<string> == <A>)",
            "exists(<A> == b)",
            "exists(<A> == g)",
            "exists(<A> == a)",
            "exists(<B> == 0)",
            "exists(<B> == 1)",
            "len(<A>)",
            "exists(<B> == 7)",
            "exists(<A> == e)",
            "exists(<A> == f)",
            "exists(<A> == d)",
            "exists(<A> == x)",
            "exists(<A> == u)",
            "exists(<A> == w)",
            "len(<string>)",
            "exists(<string> == <B>)",
            "exists(<A> == o)",
            "exists(<B> == 2)",
            "exists(<A> == m)",
            "exists(<A> == c)",
            "exists(<A> == p)",
            "exists(<start>)",
            "exists(<A> == r)",
            "exists(<string> == !ab!)",
            "exists(<B> == 4)",
            "exists(<A> == n)",
            "exists(<A> == h)",
            "exists(<A> == t)",
            "exists(<B>)",
            "exists(<A> == v)",
            "num(<B>)",
            "exists(<B> == 6)",
            "exists(<B> == 5)",
            "exists(<A> == z)",
            "exists(<string>)",
            "exists(<A> == s)",
            "exists(<B> == 3)",
            "exists(<A> == j)",
            "exists(<A> == i)",
            "exists(<B> == 8)",
            "len(<B>)",
        }

        collector = Collector(grammar, features=STANDARD_FEATURES)
        all_features = collector.get_all_features()

        all_features_repr = set([f.__repr__() for f in all_features])
        self.assertTrue(all_features_repr == expected_feature_list)

        all_features_to_string = set([str(feature) for feature in all_features])
        self.assertEqual(all_features_to_string, expected_feature_list_strings)

    def test_extract_features_length(self):
        expected_feature_list = {
            "len(<start>)",
            "len(<string>)",
            "len(<A>)",
            "len(<B>)",
        }

        collector = Collector(grammar, features={LENGTH_FEATURE})
        all_features = collector.get_all_features()
        all_features_to_string = set([str(feature) for feature in all_features])

        self.assertEqual(all_features_to_string, expected_feature_list)

    def test_parse_features_existence(self):
        expected_data = [
            (
                "9",
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ),
            (
                "3",
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            (
                "a",
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        ]

        expected_values = pd.DataFrame(
            expected_data,
            columns=[
                "input",
                "exists(<start>)",
                "exists(<start>@0)",
                "exists(<string>)",
                "exists(<string>@0)",
                "exists(<string>@1)",
                "exists(<string>@2)",
                "exists(<A>)",
                "exists(<A>@0)",
                "exists(<A>@1)",
                "exists(<A>@2)",
                "exists(<A>@3)",
                "exists(<A>@4)",
                "exists(<A>@5)",
                "exists(<A>@6)",
                "exists(<A>@7)",
                "exists(<A>@8)",
                "exists(<A>@9)",
                "exists(<A>@10)",
                "exists(<A>@11)",
                "exists(<A>@12)",
                "exists(<A>@13)",
                "exists(<A>@14)",
                "exists(<A>@15)",
                "exists(<A>@16)",
                "exists(<A>@17)",
                "exists(<A>@18)",
                "exists(<A>@19)",
                "exists(<A>@20)",
                "exists(<A>@21)",
                "exists(<A>@22)",
                "exists(<A>@23)",
                "exists(<A>@24)",
                "exists(<A>@25)",
                "exists(<B>)",
                "exists(<B>@0)",
                "exists(<B>@1)",
                "exists(<B>@2)",
                "exists(<B>@3)",
                "exists(<B>@4)",
                "exists(<B>@5)",
                "exists(<B>@6)",
                "exists(<B>@7)",
                "exists(<B>@8)",
                "exists(<B>@9)",
            ],
        )
        collector = Collector(grammar, features={EXISTENCE_FEATURE})
        input_list = ["9", "3", "a"]
        derivation_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
            for inp in input_list
        ]
        df = collector.collect_features(derivation_trees)
        self.assertTrue(df.equals(expected_values))

    def test_parse_features_numericInterpretation(self):
        expected_values = pd.DataFrame(
            [("1", 1.0), ("3", 3.0), ("a", numpy.NAN)], columns=["input", "num(<B>)"]
        )

        collector = Collector(grammar, features={NUMERIC_INTERPRETATION_FEATURE})
        input_list = ["1", "3", "a"]
        derivation_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
            for inp in input_list
        ]
        df = collector.collect_features(derivation_trees)
        self.assertTrue(df.equals(expected_values))

    def test_parse_features_length(self):
        expected_data = [
            ("1", 1.0, 1.0, numpy.NAN, 1.0),
            ("a", 1.0, 1.0, 1.0, numpy.NAN),
        ]

        expected_values = pd.DataFrame(
            expected_data,
            columns=["input", "len(<start>)", "len(<string>)", "len(<A>)", "len(<B>)"],
        )

        collector = Collector(grammar, features={LENGTH_FEATURE})
        input_list = ["1", "a"]
        derivation_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
            for inp in input_list
        ]
        df = collector.collect_features(derivation_trees)

        self.assertTrue(df.equals(expected_values))

    def test_parse_features_isDigit(self):
        expected_data = [
            ("1", True, True, numpy.NAN, True),
            ("a", False, False, False, numpy.NAN),
        ]

        expected_values = pd.DataFrame(
            expected_data,
            columns=[
                "input",
                "is_digit(<start>)",
                "is_digit(<string>)",
                "is_digit(<A>)",
                "is_digit(<B>)",
            ],
        )
        collector = Collector(grammar, features={IS_DIGIT_FEATURE})
        input_list = ["1", "a"]
        derivation_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
            for inp in input_list
        ]
        df = collector.collect_features(derivation_trees)
        self.assertTrue(df.equals(expected_values))

    def test_extract_numericInterpretation_maybe_minus(self):
        # We do not expect num(<A>) as this is not a number
        expected_feature_list = {"num(<start>)", "num(<string>)", "num(<B>)"}

        collector = Collector(
            grammar_with_maybe_minus, features={NUMERIC_INTERPRETATION_FEATURE}
        )
        all_features = collector.get_all_features()
        all_features_to_string = set([str(feature) for feature in all_features])

        self.assertEqual(all_features_to_string, expected_feature_list)

    def test_parse_numericInterpretation_maybe_minus(self):
        expected_data = [
            ("-1", -1.0, -1.0, 1.0),
            ("3", 3.0, 3.0, 3.0),
            ("-9", -9.0, -9.0, 9.0),
        ]
        expected_values = pd.DataFrame(
            expected_data,
            columns=["input", "num(<start>)", "num(<string>)", "num(<B>)"],
        )

        collector = Collector(
            grammar_with_maybe_minus, features={NUMERIC_INTERPRETATION_FEATURE}
        )
        input_list = ["-1", "3", "-9"]
        derivation_trees = [
            DerivationTree.from_parse_tree(
                next(EarleyParser(grammar_with_maybe_minus).parse(inp))
            )
            for inp in input_list
        ]
        df = collector.collect_features(derivation_trees)
        expected_values = expected_values.reindex(
            columns=sorted(expected_values.columns)
        )
        self.assertTrue(df.equals(expected_values))

    def test_parse_features_numericInterpretation_recursive(self):
        expected_values = pd.DataFrame(
            [
                ("11923", 9.0, 11923.0),
                ("3341923", 9.0, 3341923.0),
                ("9", 9.0, 9.0),
                ("a", numpy.NAN, numpy.NAN),
            ],
            columns=["input", "num(<digit>)", "num(<B>)"],
        )

        collector = Collector(grammar_rec, features={NUMERIC_INTERPRETATION_FEATURE})
        input_list = ["11923", "3341923", "9", "a"]
        derivation_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(grammar_rec).parse(inp)))
            for inp in input_list
        ]
        df = collector.collect_features(derivation_trees)

        self.assertTrue(df.equals(expected_values))

    def test_parse_features_length_recursive(self):
        expected_data = [
            ("123", 3.0, 3.0, numpy.NAN, numpy.NAN, 3.0, 1.0),
            ("ab", 2.0, 2.0, 2.0, 1.0, numpy.NAN, numpy.NAN),
        ]

        expected_values = pd.DataFrame(
            expected_data,
            columns=[
                "input",
                "len(<start>)",
                "len(<string>)",
                "len(<A>)",
                "len(<chars>)",
                "len(<B>)",
                "len(<digit>)",
            ],
        )

        collector = Collector(grammar_rec, features={LENGTH_FEATURE})
        input_list = ["123", "ab"]
        derivation_trees = [
            DerivationTree.from_parse_tree(next(EarleyParser(grammar_rec).parse(inp)))
            for inp in input_list
        ]
        df = collector.collect_features(derivation_trees)

        self.assertTrue(df.equals(expected_values))


if __name__ == "__main__":
    unittest.main()
