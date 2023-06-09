import string
import unittest
from typing import List, Dict

import numpy
from fuzzingbook.Grammars import Grammar, is_valid_grammar, srange
from fuzzingbook.Parser import EarleyParser
from isla.language import DerivationTree

from avicenna.input import Input
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
        input_list = ["9", "3", "a"]
        expected_dicts = [
            {
                "exists(<start>)": 1,
                "exists(<start>@0)": 1,
                "exists(<string>)": 1,
                "exists(<string>@0)": 0,
                "exists(<string>@1)": 1,
                "exists(<string>@2)": 0,
                "exists(<A>)": 0,
                "exists(<A>@0)": 0,
                "exists(<A>@1)": 0,
                "exists(<A>@2)": 0,
                "exists(<A>@3)": 0,
                "exists(<A>@4)": 0,
                "exists(<A>@5)": 0,
                "exists(<A>@6)": 0,
                "exists(<A>@7)": 0,
                "exists(<A>@8)": 0,
                "exists(<A>@9)": 0,
                "exists(<A>@10)": 0,
                "exists(<A>@11)": 0,
                "exists(<A>@12)": 0,
                "exists(<A>@13)": 0,
                "exists(<A>@14)": 0,
                "exists(<A>@15)": 0,
                "exists(<A>@16)": 0,
                "exists(<A>@17)": 0,
                "exists(<A>@18)": 0,
                "exists(<A>@19)": 0,
                "exists(<A>@20)": 0,
                "exists(<A>@21)": 0,
                "exists(<A>@22)": 0,
                "exists(<A>@23)": 0,
                "exists(<A>@24)": 0,
                "exists(<A>@25)": 0,
                "exists(<B>)": 1,
                "exists(<B>@0)": 0,
                "exists(<B>@1)": 0,
                "exists(<B>@2)": 0,
                "exists(<B>@3)": 0,
                "exists(<B>@4)": 0,
                "exists(<B>@5)": 0,
                "exists(<B>@6)": 0,
                "exists(<B>@7)": 0,
                "exists(<B>@8)": 0,
                "exists(<B>@9)": 1,
            },
            {
                "exists(<start>)": 1,
                "exists(<start>@0)": 1,
                "exists(<string>)": 1,
                "exists(<string>@0)": 0,
                "exists(<string>@1)": 1,
                "exists(<string>@2)": 0,
                "exists(<A>)": 0,
                "exists(<A>@0)": 0,
                "exists(<A>@1)": 0,
                "exists(<A>@2)": 0,
                "exists(<A>@3)": 0,
                "exists(<A>@4)": 0,
                "exists(<A>@5)": 0,
                "exists(<A>@6)": 0,
                "exists(<A>@7)": 0,
                "exists(<A>@8)": 0,
                "exists(<A>@9)": 0,
                "exists(<A>@10)": 0,
                "exists(<A>@11)": 0,
                "exists(<A>@12)": 0,
                "exists(<A>@13)": 0,
                "exists(<A>@14)": 0,
                "exists(<A>@15)": 0,
                "exists(<A>@16)": 0,
                "exists(<A>@17)": 0,
                "exists(<A>@18)": 0,
                "exists(<A>@19)": 0,
                "exists(<A>@20)": 0,
                "exists(<A>@21)": 0,
                "exists(<A>@22)": 0,
                "exists(<A>@23)": 0,
                "exists(<A>@24)": 0,
                "exists(<A>@25)": 0,
                "exists(<B>)": 1,
                "exists(<B>@0)": 0,
                "exists(<B>@1)": 0,
                "exists(<B>@2)": 0,
                "exists(<B>@3)": 1,
                "exists(<B>@4)": 0,
                "exists(<B>@5)": 0,
                "exists(<B>@6)": 0,
                "exists(<B>@7)": 0,
                "exists(<B>@8)": 0,
                "exists(<B>@9)": 0,
            },
            {
                "exists(<start>)": 1,
                "exists(<start>@0)": 1,
                "exists(<string>)": 1,
                "exists(<string>@0)": 1,
                "exists(<string>@1)": 0,
                "exists(<string>@2)": 0,
                "exists(<A>)": 1,
                "exists(<A>@0)": 1,
                "exists(<A>@1)": 0,
                "exists(<A>@2)": 0,
                "exists(<A>@3)": 0,
                "exists(<A>@4)": 0,
                "exists(<A>@5)": 0,
                "exists(<A>@6)": 0,
                "exists(<A>@7)": 0,
                "exists(<A>@8)": 0,
                "exists(<A>@9)": 0,
                "exists(<A>@10)": 0,
                "exists(<A>@11)": 0,
                "exists(<A>@12)": 0,
                "exists(<A>@13)": 0,
                "exists(<A>@14)": 0,
                "exists(<A>@15)": 0,
                "exists(<A>@16)": 0,
                "exists(<A>@17)": 0,
                "exists(<A>@18)": 0,
                "exists(<A>@19)": 0,
                "exists(<A>@20)": 0,
                "exists(<A>@21)": 0,
                "exists(<A>@22)": 0,
                "exists(<A>@23)": 0,
                "exists(<A>@24)": 0,
                "exists(<A>@25)": 0,
                "exists(<B>)": 0,
                "exists(<B>@0)": 0,
                "exists(<B>@1)": 0,
                "exists(<B>@2)": 0,
                "exists(<B>@3)": 0,
                "exists(<B>@4)": 0,
                "exists(<B>@5)": 0,
                "exists(<B>@6)": 0,
                "exists(<B>@7)": 0,
                "exists(<B>@8)": 0,
                "exists(<B>@9)": 0,
            },
        ]

        test_inputs = [
            Input(
                DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
            )
            for inp in input_list
        ]

        collector = Collector(grammar, features={EXISTENCE_FEATURE})

        for inp, expected in zip(test_inputs, expected_dicts):
            result = collector.collect_features(inp)
            self.assertTrue(result == expected)

    def test_parse_features_numericInterpretation(self):
        input_list = ["1", "3", "a"]
        expected_dicts: List[Dict] = [
            {"num(<B>)": 1.0},
            {"num(<B>)": 3.0},
            {"num(<B>)": numpy.NAN},
        ]
        collector = Collector(grammar, features={NUMERIC_INTERPRETATION_FEATURE})
        test_inputs = [
            Input(
                tree=DerivationTree.from_parse_tree(
                    next(EarleyParser(grammar).parse(inp))
                )
            )
            for inp in input_list
        ]
        for inp, expected in zip(test_inputs, expected_dicts):
            result = collector.collect_features(inp)
            self.assertTrue(result == expected)

    def test_parse_features_length(self):
        input_list = ["1", "a"]
        expected_dicts = [
            {
                "len(<start>)": 1.0,
                "len(<string>)": 1.0,
                "len(<A>)": numpy.NAN,
                "len(<B>)": 1.0,
            },
            {
                "len(<start>)": 1.0,
                "len(<string>)": 1.0,
                "len(<A>)": 1.0,
                "len(<B>)": numpy.NAN,
            },
        ]
        collector = Collector(grammar, features={LENGTH_FEATURE})
        test_inputs = [
            Input(
                tree=DerivationTree.from_parse_tree(
                    next(EarleyParser(grammar).parse(inp))
                )
            )
            for inp in input_list
        ]
        for inp, expected in zip(test_inputs, expected_dicts):
            result = collector.collect_features(inp)
            self.assertTrue(result == expected)

    def test_parse_features_isDigit(self):
        input_list = ["1", "a"]
        expected_dicts = [
            {
                "is_digit(<start>)": True,
                "is_digit(<string>)": True,
                "is_digit(<A>)": numpy.NAN,
                "is_digit(<B>)": True,
            },
            {
                "is_digit(<start>)": False,
                "is_digit(<string>)": False,
                "is_digit(<A>)": False,
                "is_digit(<B>)": numpy.NAN,
            },
        ]

        collector = Collector(grammar, features={IS_DIGIT_FEATURE})
        test_inputs = [
            Input(
                tree=DerivationTree.from_parse_tree(
                    next(EarleyParser(grammar).parse(inp))
                )
            )
            for inp in input_list
        ]
        for inp, expected in zip(test_inputs, expected_dicts):
            result = collector.collect_features(inp)
            self.assertTrue(result == expected)

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
        input_list = ["-1", "3", "-9"]
        expected_dicts = [
            {"num(<start>)": -1.0, "num(<string>)": -1.0, "num(<B>)": 1.0},
            {"num(<start>)": 3.0, "num(<string>)": 3.0, "num(<B>)": 3.0},
            {"num(<start>)": -9.0, "num(<string>)": -9.0, "num(<B>)": 9.0},
        ]

        collector = Collector(
            grammar_with_maybe_minus, features={NUMERIC_INTERPRETATION_FEATURE}
        )
        test_inputs = [
            Input(
                tree=DerivationTree.from_parse_tree(
                    next(EarleyParser(grammar_with_maybe_minus).parse(inp))
                )
            )
            for inp in input_list
        ]
        for inp, expected in zip(test_inputs, expected_dicts):
            result = collector.collect_features(inp)
            self.assertTrue(result == expected)

    def test_parse_features_numericInterpretation_recursive(self):
        input_list = ["11923", "3341923", "9", "a"]
        expected_dicts = [
            {"num(<digit>)": 9.0, "num(<B>)": 11923.0},
            {"num(<digit>)": 9.0, "num(<B>)": 3341923.0},
            {"num(<digit>)": 9.0, "num(<B>)": 9.0},
            {"num(<digit>)": numpy.NAN, "num(<B>)": numpy.NAN},
        ]

        collector = Collector(grammar_rec, features={NUMERIC_INTERPRETATION_FEATURE})
        test_inputs = [
            Input(
                tree=DerivationTree.from_parse_tree(
                    next(EarleyParser(grammar_rec).parse(inp))
                )
            )
            for inp in input_list
        ]
        for inp, expected in zip(test_inputs, expected_dicts):
            result = collector.collect_features(inp)
            self.assertTrue(result == expected)

    def test_parse_features_length_recursive(self):
        input_list = ["123", "ab"]
        expected_dicts = [
            {
                "len(<start>)": 3.0,
                "len(<string>)": 3.0,
                "len(<A>)": numpy.NAN,
                "len(<chars>)": numpy.NAN,
                "len(<B>)": 3.0,
                "len(<digit>)": 1.0,
            },
            {
                "len(<start>)": 2.0,
                "len(<string>)": 2.0,
                "len(<A>)": 2.0,
                "len(<chars>)": 1.0,
                "len(<B>)": numpy.NAN,
                "len(<digit>)": numpy.NAN,
            },
        ]

        collector = Collector(grammar_rec, features={LENGTH_FEATURE})
        test_inputs = [
            Input(
                tree=DerivationTree.from_parse_tree(
                    next(EarleyParser(grammar_rec).parse(inp))
                )
            )
            for inp in input_list
        ]
        for inp, expected in zip(test_inputs, expected_dicts):
            result = collector.collect_features(inp)
            self.assertTrue(result == expected)


if __name__ == "__main__":
    unittest.main()
