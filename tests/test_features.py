import string
import unittest

from numpy import inf
from fuzzingbook.Grammars import Grammar, is_valid_grammar, srange

from avicenna.input import Input
from avicenna.feature_collector import (
    FeatureFactory,
    ExistenceFeature,
    DerivationFeature,
    NumericFeature,
    LengthFeature,
    GrammarFeatureCollector,
)

grammar: Grammar = {
    "<start>": ["<string>"],
    "<string>": ["<A>", "<B>", "!ab!"],
    "<A>": srange(string.ascii_lowercase[:5]),
    "<B>": srange(string.digits[:5]),
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
    def test_build_existence_feature(self):
        expected_feature_list = [
            ExistenceFeature("<start>"),
            ExistenceFeature("<string>"),
            ExistenceFeature("<A>"),
            ExistenceFeature("<B>"),
        ]

        factory = FeatureFactory(grammar)
        features = factory.build([ExistenceFeature])

        self.assertEqual(features, expected_feature_list)

    def test_build_derivation_feature(self):
        expected_feature_list = [
            DerivationFeature("<start>", "<string>"),
            DerivationFeature("<string>", "<A>"),
            DerivationFeature("<string>", "<B>"),
            DerivationFeature("<string>", "!ab!"),
            DerivationFeature("<A>", "a"),
            DerivationFeature("<A>", "b"),
            DerivationFeature("<A>", "c"),
            DerivationFeature("<A>", "d"),
            DerivationFeature("<A>", "e"),
            DerivationFeature("<B>", "0"),
            DerivationFeature("<B>", "1"),
            DerivationFeature("<B>", "2"),
            DerivationFeature("<B>", "3"),
            DerivationFeature("<B>", "4"),
        ]

        factory = FeatureFactory(grammar)
        features = factory.build([DerivationFeature])

        self.assertEqual(features, expected_feature_list)

    def test_build_numeric_feature(self):
        expected_feature_list = [
            NumericFeature("<B>"),
        ]

        factory = FeatureFactory(grammar)
        features = factory.build([NumericFeature])

        self.assertEqual(features, expected_feature_list)

    def test_build_numeric_feature_maybe(self):
        # We do not expect num(<A>) as this is not a number
        expected_feature_list = [
            NumericFeature("<start>"),
            NumericFeature("<string>"),
            NumericFeature("<B>"),
        ]

        factory = FeatureFactory(grammar_with_maybe_minus)
        features = factory.build([NumericFeature])

        self.assertEqual(set(features), set(expected_feature_list))

    def test_build_length_feature(self):
        expected_feature_list = [
            LengthFeature("<start>"),
            LengthFeature("<string>"),
            LengthFeature("<A>"),
            LengthFeature("<B>"),
        ]

        factory = FeatureFactory(grammar_with_maybe_minus)
        features = factory.build([LengthFeature])

        self.assertEqual(features, expected_feature_list)

    def test_parse_existence_feature(self):
        inputs = ["4", "3", "a"]
        test_inputs = [Input.from_str(grammar, inp) for inp in inputs]

        expected_feature_vectors = [
            {
                ExistenceFeature("<start>"): 1,
                ExistenceFeature("<string>"): 1,
                ExistenceFeature("<A>"): 0,
                ExistenceFeature("<B>"): 1,
            },
            {
                ExistenceFeature("<start>"): 1,
                ExistenceFeature("<string>"): 1,
                ExistenceFeature("<A>"): 0,
                ExistenceFeature("<B>"): 1,
            },
            {
                ExistenceFeature("<start>"): 1,
                ExistenceFeature("<string>"): 1,
                ExistenceFeature("<A>"): 1,
                ExistenceFeature("<B>"): 0,
            },
        ]

        collector = GrammarFeatureCollector(grammar, [ExistenceFeature])
        for test_input, expected_feature_vectors in zip(
            test_inputs, expected_feature_vectors
        ):
            feature_vector = collector.collect_features(test_input)
            self.assertEqual(feature_vector.features, expected_feature_vectors)

    def test_parse_numeric_feature(self):
        inputs = ["4", "3", "a"]
        test_inputs = [Input.from_str(grammar, inp) for inp in inputs]

        expected_feature_vectors = [
            {
                NumericFeature("<B>"): 4.0,
            },
            {
                NumericFeature("<B>"): 3.0,
            },
            {
                NumericFeature("<B>"): -inf,
            },
        ]

        collector = GrammarFeatureCollector(grammar, [NumericFeature])
        for test_input, expected_feature_vectors in zip(
            test_inputs, expected_feature_vectors
        ):
            feature_vector = collector.collect_features(test_input)
            self.assertEqual(feature_vector.features, expected_feature_vectors)

    def test_parse_features_length(self):
        inputs = ["4", "3", "a"]
        test_inputs = [Input.from_str(grammar, inp) for inp in inputs]

        expected_feature_vectors = [
            {
                LengthFeature("<start>"): 1,
                LengthFeature("<string>"): 1,
                LengthFeature("<A>"): 0,
                LengthFeature("<B>"): 1,
            },
            {
                LengthFeature("<start>"): 1,
                LengthFeature("<string>"): 1,
                LengthFeature("<A>"): 0,
                LengthFeature("<B>"): 1,
            },
            {
                LengthFeature("<start>"): 1,
                LengthFeature("<string>"): 1,
                LengthFeature("<A>"): 1,
                LengthFeature("<B>"): 0,
            },
        ]

        collector = GrammarFeatureCollector(grammar, [LengthFeature])
        for test_input, expected_feature_vectors in zip(
            test_inputs, expected_feature_vectors
        ):
            feature_vector = collector.collect_features(test_input)
            self.assertEqual(feature_vector.features, expected_feature_vectors)

    def test_parse_numericInterpretation_maybe_minus(self):
        inputs = ["-1", "3", "-9"]
        test_inputs = [Input.from_str(grammar_with_maybe_minus, inp) for inp in inputs]

        expected_feature_vectors = [
            {
                NumericFeature("<start>"): -1.0,
                NumericFeature("<string>"): -1.0,
                NumericFeature("<B>"): 1.0,
            },
            {
                NumericFeature("<start>"): 3.0,
                NumericFeature("<string>"): 3.0,
                NumericFeature("<B>"): 3.0,
            },
            {
                NumericFeature("<start>"): -9.0,
                NumericFeature("<string>"): -9.0,
                NumericFeature("<B>"): 9.0,
            },
        ]

        collector = GrammarFeatureCollector(grammar_with_maybe_minus, [NumericFeature])
        for test_input, expected_feature_vectors in zip(
            test_inputs, expected_feature_vectors
        ):
            feature_vector = collector.collect_features(test_input)
            self.assertEqual(feature_vector.features, expected_feature_vectors)

    def test_parse_features_numericInterpretation_recursive(self):
        inputs = ["11923", "3341923", "9", "a"]
        test_inputs = [Input.from_str(grammar_rec, inp) for inp in inputs]

        expected_feature_vectors = [
            {
                NumericFeature("<digit>"): 9.0,
                NumericFeature("<B>"): 11923.0,
            },
            {
                NumericFeature("<digit>"): 9.0,
                NumericFeature("<B>"): 3341923.0,
            },
            {
                NumericFeature("<digit>"): 9.0,
                NumericFeature("<B>"): 9.0,
            },
            {
                NumericFeature("<digit>"): -inf,
                NumericFeature("<B>"): -inf,
            },
        ]

        collector = GrammarFeatureCollector(grammar_rec, [NumericFeature])
        for test_input, expected_feature_vectors in zip(
            test_inputs, expected_feature_vectors
        ):
            feature_vector = collector.collect_features(test_input)
            self.assertEqual(feature_vector.features, expected_feature_vectors)

    def test_parse_features_length_recursive(self):
        inputs = ["123", "ab"]
        test_inputs = [Input.from_str(grammar_rec, inp) for inp in inputs]

        expected_feature_vectors = [
            {
                LengthFeature("<start>"): 3,
                LengthFeature("<string>"): 3,
                LengthFeature("<A>"): 0,
                LengthFeature("<chars>"): 0,
                LengthFeature("<B>"): 3,
                LengthFeature("<digit>"): 1,
            },
            {
                LengthFeature("<start>"): 2,
                LengthFeature("<string>"): 2,
                LengthFeature("<A>"): 2,
                LengthFeature("<chars>"): 1,
                LengthFeature("<B>"): 0,
                LengthFeature("<digit>"): 0,
            }
        ]

        collector = GrammarFeatureCollector(grammar_rec, [LengthFeature])
        for test_input, expected_feature_vectors in zip(
            test_inputs, expected_feature_vectors
        ):
            feature_vector = collector.collect_features(test_input)
            self.assertEqual(feature_vector.features, expected_feature_vectors)


if __name__ == "__main__":
    unittest.main()
