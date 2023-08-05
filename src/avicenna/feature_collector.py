from functools import lru_cache
from typing import List, Set, Dict, Optional, Any
import re
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy
import pandas
from fuzzingbook.Grammars import is_nonterminal, Grammar, reachable_nonterminals
from isla.language import DerivationTree

from avicenna.features import FeatureWrapper, STANDARD_FEATURES
from avicenna.input import Input
from avicenna.oracle import OracleResult


class Feature(ABC):
    def __init__(self, non_terminal: str, key: str):
        self.non_terminal = non_terminal
        self.key = key

    def __repr__(self) -> str:
        return f"{self.non_terminal} -> {self.key}"

    @abstractmethod
    def default_value(self):
        raise NotImplementedError()

    @abstractmethod
    def type(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, subtree: DerivationTree) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def factory_method(cls, grammar):
        raise NotImplementedError


class ExistenceFeature(Feature):
    def __repr__(self) -> str:
        return (
            f"exists({self.non_terminal})"
            if self.non_terminal == self.key
            else f"exists({self.non_terminal} -> {self.key})"
        )

    @property
    def default_value(self):
        return 0

    @property
    def type(self):
        return int

    def evaluate(self, subtree: DerivationTree) -> Any:
        current_node, children = subtree

        if self.non_terminal == current_node and self.key == current_node:
            return 1
        else:
            expansion = "".join([child[0] for child in children])
            if self.key == expansion:
                return 1

    @classmethod
    def factory_method(cls, grammar) -> List[Feature]:
        features = []

        for non_terminal in grammar:
            features.append(cls(non_terminal, non_terminal))
            # add all alternatives
            for expansion in grammar[non_terminal]:
                features.append(cls(non_terminal, expansion))

        return features


class NumericFeature(Feature):
    def __repr__(self):
        return f"num({self.non_terminal})"

    @property
    def default_value(self):
        return numpy.NAN

    @property
    def type(self):
        return float

    def evaluate(self, subtree: DerivationTree) -> Any:
        try:
            value = float(tree_to_string(subtree))
            return value
        except ValueError:
            return self.default_value()

    @classmethod
    def factory_method(cls, grammar) -> List[Feature]:
        derivable_chars = cls.get_derivable_chars(grammar)
        return cls.get_features(derivable_chars)

    @classmethod
    def get_derivable_chars(cls, grammar: Grammar) -> Dict[str, Set[str]]:
        """
        Gets all the derivable characters for each rule in the grammar.
        :param grammar: The input grammar.
        :return: A mapping from each rule to a set of derivable characters.
        """
        # Regex for non-terminal symbols in expansions
        re_non_terminal = re.compile(r"(<[^<> ]*>)")

        # Mapping from non-terminals to derivable terminal chars
        derivable_chars = defaultdict(set)

        # Populate initial derivable_chars
        for rule, expansions in grammar.items():
            for expansion in expansions:
                # Remove non-terminal symbols and whitespace from expansion
                terminals = re.sub(re_non_terminal, "", expansion)
                # Add each terminal char to the set of derivable chars
                for char in terminals:
                    derivable_chars[rule].add(char)

        # Update derivable_chars until convergence
        while True:
            if not cls.update_derivable_chars(grammar, derivable_chars):
                break

        return derivable_chars

    @classmethod
    def update_derivable_chars(
        cls, grammar: Grammar, derivable_chars: Dict[str, Set[str]]
    ) -> bool:
        """
        Update the mapping of derivable characters for each rule.
        :param grammar: The input grammar.
        :param derivable_chars: The existing mapping of derivable characters.
        :return: True if any set of derivable chars was updated, False otherwise.
        """
        updated = False
        for rule in grammar:
            for reachable_rule in reachable_nonterminals(grammar, rule):
                before_update = len(derivable_chars[rule])
                derivable_chars[rule].update(derivable_chars[reachable_rule])
                after_update = len(derivable_chars[rule])

                # Set of derivable chars was updated
                if after_update > before_update:
                    updated = True
        return updated

    @classmethod
    def get_features(cls, derivable_chars: Dict[str, Set[str]]) -> List[Feature]:
        """
        Gets a list of NumericInterpretation features for each rule that derives only numeric characters.
        :param derivable_chars: The mapping of derivable characters.
        :return: A list of NumericInterpretation features.
        """
        features = []
        numeric_chars = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
        numeric_symbols = {".", "-"}

        for rule, chars in derivable_chars.items():
            non_numeric_chars = (chars - numeric_chars) - numeric_symbols
            has_numeric_chars = len(chars.intersection(numeric_chars)) > 0

            # Rule derives only numeric characters and at least one numeric character
            if len(non_numeric_chars) == 0 and has_numeric_chars:
                features.append(cls(f"num({rule})", rule))

        return features


class LengthFeature(Feature):
    def __repr__(self):
        return f"len({self.key})"

    @property
    def default_value(self):
        return numpy.NAN

    def type(self):
        return int

    def evaluate(self, subtree: DerivationTree) -> Any:
        return len(tree_to_string(subtree))

    @classmethod
    def factory_method(cls, grammar) -> List[Feature]:
        features = []
        for non_terminal in grammar:
            features.append(cls(non_terminal, non_terminal))
        return features


class FeatureFactory:
    def __init__(self, grammar):
        self.grammar = grammar

    def build(self, feature_types=None) -> List[Feature]:
        if feature_types is None:
            feature_types = [ExistenceFeature, NumericFeature]

        all_features = list()
        for feature_type in feature_types:
            all_features.extend(feature_type.factory_method(self.grammar))
        return all_features


class FeatureVector:
    def __init__(
        self,
        result: Optional[OracleResult] = None,
    ):
        self.result = result
        self.features: Dict[Feature, Any] = dict()

    def get_feature_value(self, feature: Feature) -> Any:
        if feature in self.features:
            return self.features[feature]
        else:
            return feature.default_value

    def set_feature(self, feature: Feature, value: any):
        self.features[feature] = max(value, self.features[feature])

    def get_features(self) -> Dict[Feature, Any]:
        return self.features

    def __repr__(self):
        return f"{self.result.name}{self.features}"

    def __str__(self):
        return f"{self.result.name}{self.features}"


class FeatureCollector(ABC):
    def __init__(
        self, grammar: Grammar, feature_types: Set[FeatureWrapper] = STANDARD_FEATURES
    ):
        self.grammar = grammar
        self.features = self.construct_features(feature_types)

    def construct_features(self, feature_types: Set[FeatureWrapper]) -> List[Feature]:
        return [
            feature
            for feature_class in feature_types
            for feature in feature_class.extract_from_grammar(grammar=self.grammar)
        ]

    @abstractmethod
    def collect_features(self, test_input: Input) -> Dict[str, Any]:
        pass


class GrammarFeatureCollector(FeatureCollector):
    def collect_features_from_list(self, test_inputs: Set[Input]) -> pandas.DataFrame:
        data = [self.collect_features(inp) for inp in test_inputs]
        return pandas.DataFrame.from_records(data)

    def collect_features(self, test_input: Input) -> FeatureVector:
        feature_vector = FeatureVector(test_input.oracle)
        self.set_features(test_input.tree, feature_vector)
        return feature_vector

    def set_features(self, tree: DerivationTree, feature_vector: FeatureVector):
        (node, children) = tree

        corresponding_features_1d = self.get_corresponding_feature(tree)

        for corresponding_feature in corresponding_features_1d:
            value = corresponding_feature.evaluate(tree)
            feature_vector.set_feature(corresponding_feature, value)

        for child in children:
            if is_nonterminal(child[0]):
                self.set_features(child, feature_vector)

    @lru_cache
    def get_corresponding_feature(self, tree: DerivationTree) -> List[Feature]:
        current_node, children = tree
        expansion = "".join(child[0] for child in children)

        return [
            feature
            for feature in self.features
            if (
                feature.non_terminal == current_node
                and (feature.key == current_node or feature.key == expansion)
            )
        ]


def tree_to_string(tree: DerivationTree) -> str:
    symbol, children, *_ = tree
    if children:
        return "".join(tree_to_string(c) for c in children)
    else:
        return "" if is_nonterminal(symbol) else symbol
