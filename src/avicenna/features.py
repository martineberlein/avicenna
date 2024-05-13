from typing import List, Set, Dict, Optional, Any
import re
from abc import ABC, abstractmethod
from collections import defaultdict

from debugging_framework.input.oracle import OracleResult

from fuzzingbook.Grammars import is_nonterminal, Grammar, reachable_nonterminals
from isla.language import DerivationTree


class Feature(ABC):
    def __init__(self, non_terminal: str):
        self.non_terminal = non_terminal

    def __repr__(self) -> str:
        return (
            self._repr()
            .replace('"', "&quot;")
            .replace(",", "&comma;")
            .replace("[", "&lsqb;")
            .replace("]", "&rsqb;")
            .replace("{", "&lcub;")
            .replace("}", "&rcub;")
            .replace(":", "&colon;")
            .replace("'", "&#39;")
            .replace(" ", "&nbsp;")
            .replace("\n", "&#13;")
            #.replace("\r", "&#60;")
            .replace("\\", "")
        )

    @abstractmethod
    def _repr(self) -> str:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.__hash__() == hash(other)
        return False

    @abstractmethod
    def default_value(self):
        raise NotImplementedError

    @abstractmethod
    def type(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, subtree: DerivationTree) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def factory_method(cls, grammar):
        raise NotImplementedError


class ExistenceFeature(Feature):
    def __init__(self, non_terminal: str):
        super().__init__(non_terminal)

    def _repr(self) -> str:
        return f"exists({self.non_terminal})"

    @property
    def default_value(self):
        return 0

    @property
    def type(self):
        return int

    def evaluate(self, subtree: DerivationTree) -> int:
        current_node, _ = subtree
        return int(self.non_terminal == current_node)

    @classmethod
    def factory_method(cls, grammar) -> List[Feature]:
        return [cls(non_terminal) for non_terminal in grammar]


class DerivationFeature(Feature):
    def __init__(self, non_terminal: str, expansion: str):
        super().__init__(non_terminal)
        self.expansion = expansion

    def _repr(self) -> str:
        return f"exists({self.non_terminal} -> {self.expansion})"

    @property
    def default_value(self):
        return 0

    @property
    def type(self):
        return int

    def evaluate(self, subtree: DerivationTree) -> int:
        current_node, children = subtree

        expansion = "".join([child[0] for child in children])
        return int(self.non_terminal == current_node and self.expansion == expansion)

    @classmethod
    def factory_method(cls, grammar) -> List[Feature]:
        features = []

        for non_terminal in grammar:
            for expansion in grammar[non_terminal]:
                features.append(cls(non_terminal, expansion))

        return features


class NumericFeature(Feature):
    def _repr(self):
        return f"num({self.non_terminal})"

    @property
    def default_value(self):
        return float("-inf")

    @property
    def type(self):
        return float

    def evaluate(self, subtree: DerivationTree) -> Any:
        try:
            value = float(tree_to_string(subtree))
            return value
        except ValueError:
            return self.default_value

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

        for non_terminal, chars in derivable_chars.items():
            non_numeric_chars = (chars - numeric_chars) - numeric_symbols
            has_numeric_chars = len(chars.intersection(numeric_chars)) > 0

            # Rule derives only numeric characters and at least one numeric character
            if len(non_numeric_chars) == 0 and has_numeric_chars:
                features.append(cls(non_terminal))

        return features


class LengthFeature(Feature):
    def _repr(self):
        return f"len({self.non_terminal})"

    @property
    def default_value(self):
        return 0

    def type(self):
        return int

    def evaluate(self, subtree: DerivationTree) -> Any:
        return len(tree_to_string(subtree))

    @classmethod
    def factory_method(cls, grammar) -> List[Feature]:
        features = []
        for non_terminal in grammar:
            features.append(cls(non_terminal))
        return features


class FeatureFactory:
    def __init__(self, grammar):
        self.grammar = grammar

    def build(self, feature_types=None) -> List[Feature]:
        if feature_types is None:
            feature_types = [
                ExistenceFeature,
                DerivationFeature,
                NumericFeature,
                LengthFeature,
            ]

        all_features = list()
        for feature_type in feature_types:
            all_features.extend(feature_type.factory_method(self.grammar))
        return all_features


class FeatureVector:
    def __init__(
        self,
        test_input: str,
        result: Optional[OracleResult] = None,
    ):
        self.test_input = test_input
        self.result = result
        self.features: Dict[Feature, Any] = dict()

    def get_feature_value(self, feature: Feature) -> Any:
        if feature in self.features:
            return self.features[feature]
        else:
            return feature.default_value

    def set_feature(self, feature: Feature, value: any):
        if feature in self.features.keys():
            self.features[feature] = max(value, self.features[feature])
        else:
            self.features[feature] = value

    def get_features(self) -> Dict[Feature, Any]:
        return self.features

    def __repr__(self):
        return f"{self.test_input}: {self.features}"


def tree_to_string(tree: DerivationTree) -> str:
    symbol, children, *_ = tree
    if children:
        return "".join(tree_to_string(c) for c in children)
    else:
        return "" if is_nonterminal(symbol) else symbol
