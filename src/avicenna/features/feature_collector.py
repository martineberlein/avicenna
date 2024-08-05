from functools import lru_cache
from typing import List, Dict, Optional, Any, Type
from abc import ABC, abstractmethod

from fuzzingbook.Grammars import is_nonterminal, Grammar
from isla.language import DerivationTree

from avicenna.input.input import Input
from avicenna.features.features import (
    ExistenceFeature,
    DerivationFeature,
    NumericFeature,
    LengthFeature,
    Feature,
    FeatureVector,
    FeatureFactory,
)

DEFAULT_FEATURE_TYPES: List[Type[Feature]] = [
    ExistenceFeature,
    DerivationFeature,
    NumericFeature,
    LengthFeature,
]


class FeatureCollector(ABC):
    def __init__(
        self, grammar: Grammar, feature_types: Optional[List[Type[Feature]]] = None
    ):
        self.grammar = grammar
        feature_types = feature_types if feature_types else DEFAULT_FEATURE_TYPES
        self.features = self.construct_features(feature_types)

    def construct_features(self, feature_types: List[Type[Feature]]) -> List[Feature]:
        factory = FeatureFactory(self.grammar)
        return factory.build(feature_types)

    @abstractmethod
    def collect_features(self, test_input: Input) -> Dict[str, Any]:
        pass


class GrammarFeatureCollector(FeatureCollector):
    def collect_features(self, test_input: Input) -> FeatureVector:
        feature_vector = FeatureVector(str(test_input))

        for feature in self.features:
            feature_vector.set_feature(feature, feature.default_value)

        self.set_features(test_input.tree, feature_vector)
        return feature_vector

    def set_features(self, tree: DerivationTree, feature_vector: FeatureVector):
        (node, children) = tree

        corresponding_features_1d = self.get_corresponding_feature(node)

        for corresponding_feature in corresponding_features_1d:
            value = corresponding_feature.evaluate(tree)
            feature_vector.set_feature(corresponding_feature, value)

        for child in children:
            if is_nonterminal(child[0]):
                self.set_features(child, feature_vector)

    @lru_cache
    def get_corresponding_feature(self, current_node: str) -> List[Feature]:
        return [
            feature
            for feature in self.features
            if (feature.non_terminal == current_node)
        ]
