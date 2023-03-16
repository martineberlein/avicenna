import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Optional, Any, Callable

import numpy
from fuzzingbook.GrammarFuzzer import tree_to_string
from fuzzingbook.Grammars import Grammar
from fuzzingbook.Grammars import reachable_nonterminals
from numpy import isnan

DerivationTree = Tuple[str, Optional[List[Any]]]


class FeatureWrapper:
    def __init__(self, dummy_feature, extract_grammar: Callable):
        self.feature: Callable = dummy_feature
        self.extract_from_grammar = extract_grammar


class Feature(ABC):
    """
    The abstract base class for grammar features.

    Args:
        name : A unique identifier name for this feature.
        rule : The production rule (e.g., '<function>' or '<value>').
        key  : The feature key (e.g., the chosen alternative or rule itself).
    """

    def __init__(self, name: str, rule: str, key: str) -> None:
        self.name = name
        self.rule = rule
        self.key = key
        super().__init__()

    def __repr__(self) -> str:
        """Returns a printable string representation of the feature."""
        return self.name

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def initialization_value(self):
        """
        Returns the initialization value to instantiate the feature table.
        """
        pass

    @abstractmethod
    def evaluate(self, derivation_tree, feature_table):
        """Returns the feature value for a given derivation tree of an input."""
        pass


class ExistenceFeature(Feature):
    """
    This class represents existence features of a grammar. Existence features indicate
    whether a particular production rule was used in the derivation sequence of an input.
    For a given production rule P -> A | B, a production existence feature for P and
    alternative existence features for each alternative (i.e., A and B) are defined.

    name : A unique identifier name for this feature.
    rule : The production rule.
    key  : The feature key, equal to the rule attribute for production features,
           or equal to the corresponding alternative for alternative features.
    """

    def __init__(self, name: str, rule: str, key: str) -> None:
        super().__init__(name, rule, key)

    def __str__(self):
        if self.rule == self.key:
            return f"exists({self.rule})"
        else:
            return f"exists({self.rule} == {self.key})"

    def initialization_value(self):
        return 0

    def evaluate(self, derivation_tree, feature_table):
        (node, children) = derivation_tree
        # When this feature is called it exists in the tree.
        if self.rule == node and self.key == node:
            return 1
        else:
            expansion = "".join([child[0] for child in children])
            if self.key == expansion:
                return 1
                #  TODO What happens when A-> A ?? Is this a problem?
        raise AssertionError(
            "This state should not be reachable. Feature evaluation does not work."
        )


class NumericInterpretation(Feature):
    """
    This class represents numeric interpretation features of a grammar. These features
    are defined for productions that only derive words composed of the characters
    [0-9], '.', and '-'. The returned feature value corresponds to the maximum
    floating-point number interpretation of the derived words of a production.

    name : A unique identifier name for this feature. Should not contain Whitespaces.
           e.g., 'num(<integer>)'
    rule : The production rule.
    """

    def __init__(self, name: str, rule: str) -> None:
        super().__init__(name, rule, rule)

    def __str__(self) -> str:
        return f"num({self.key})"

    def initialization_value(self):
        return numpy.NAN

    def evaluate(self, derivation_tree, feature_table):
        try:
            value = float(tree_to_string(derivation_tree))
            logging.debug(f"{self.name} has feature-value length: {value}")
            logging.debug(
                f"Feature table at feature {self.name} has value {feature_table[self.name]}"
                f" of type {type(feature_table[self.name])}"
            )
            if feature_table[self.name] < value or isnan(feature_table[self.name]):
                logging.debug(
                    f"Replacing feature-value {feature_table[self.name]} with {value}"
                )
                return value
        except ValueError:
            pass


class LengthFeature(Feature):
    def __init__(self, name: str, rule: str) -> None:
        super().__init__(name, rule, rule)

    def __str__(self) -> str:
        return f"len({self.key})"

    def initialization_value(self):
        return numpy.NAN

    def evaluate(self, derivation_tree, feature_table):
        try:
            value = len(tree_to_string(derivation_tree))
            logging.debug(
                f"{self.name} has feature-value length: {len(tree_to_string(derivation_tree))}"
            )
            logging.debug(
                f"Feature table at feature {self.name} has value {feature_table[self.name]}"
                f" of type {type(feature_table[self.name])}"
            )
            if feature_table[self.name] < value or isnan(feature_table[self.name]):
                logging.debug(
                    f"Replacing feature-value {feature_table[self.name]} with {value}"
                )
                return float(value)
        except ValueError:
            pass


class IsDigitFeature(Feature):
    def __init__(self, name: str, rule: str) -> None:
        super().__init__(name, rule, rule)

    def __str__(self) -> str:
        return f"is_digit({self.key})"

    def initialization_value(self):
        return numpy.NAN

    def evaluate(self, derivation_tree, feature_table):
        logging.debug(
            f"{self.name} evaluating is_digit for: {tree_to_string(derivation_tree)}"
        )
        try:
            if isinstance(int(tree_to_string(derivation_tree)), int):
                logging.debug(
                    f"{self.name} is_digit is true fpr: {int(tree_to_string(derivation_tree))}"
                )
                return True
        except ValueError:
            pass

        logging.debug(
            f"{self.name} is not true for: {len(tree_to_string(derivation_tree))}"
        )
        return False


def extract_existence(grammar: Grammar) -> List[Feature]:
    """
    Extracts all existence features from the grammar and returns them as a list.
    grammar : The input grammar.
    """

    features = []

    for rule in grammar:
        # add the rule
        features.append(ExistenceFeature(f"exists({rule})", rule, rule))
        # add all alternatives
        for count, expansion in enumerate(grammar[rule]):
            features.append(
                ExistenceFeature(f"exists({rule}@{count})", rule, expansion)
            )

    return features


EXISTENCE_FEATURE = FeatureWrapper(ExistenceFeature, extract_existence)

# Regex for non-terminal symbols in expansions
RE_NONTERMINAL = re.compile(r"(<[^<> ]*>)")


def extract_numeric(grammar: Grammar) -> List[Feature]:
    """Extracts all numeric interpretation features from the grammar and returns them as a list.
    grammar : The input grammar.
    """

    features = []

    # Mapping from non-terminals to derivable terminal chars
    derivable_chars = defaultdict(set)

    for rule in grammar:
        for expansion in grammar[rule]:

            # Remove non-terminal symbols and whitespace from expansion
            terminals = re.sub(RE_NONTERMINAL, "", expansion)  # .replace(' ', '')

            # Add each terminal char to the set of derivable chars
            for c in terminals:
                derivable_chars[rule].add(c)

    # Repeatedly update the mapping until convergence
    while True:
        updated = False
        for rule in grammar:
            for r in reachable_nonterminals(grammar, rule):
                before = len(derivable_chars[rule])
                derivable_chars[rule].update(derivable_chars[r])
                after = len(derivable_chars[rule])

                # Set of derivable chars was updated
                if after > before:
                    updated = True

        if not updated:
            break

    numeric_chars = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    numeric_symbols = {".", "-"}

    for key in derivable_chars:
        # Check if derivable chars contain only numeric numbers
        # and check if at least one number is in the set of derivable chars
        if (
            len((derivable_chars[key] - numeric_chars) - numeric_symbols) == 0
            and len(derivable_chars[key].intersection(numeric_chars)) > 0
        ):
            features.append(NumericInterpretation(f"num({key})", key))

    return features


NUMERIC_INTERPRETATION_FEATURE = FeatureWrapper(NumericInterpretation, extract_numeric)


def extract_length(grammar: Grammar) -> List[Feature]:
    features = []
    for rule in grammar:
        features.append(LengthFeature(f"len({rule})", rule))
    return features


LENGTH_FEATURE = FeatureWrapper(LengthFeature, extract_length)


def extract_is_digit(grammar: Grammar) -> List[Feature]:
    features = []
    for rule in grammar:
        features.append(IsDigitFeature(f"is_digit({rule})", rule))
    return features


IS_DIGIT_FEATURE = FeatureWrapper(IsDigitFeature, extract_is_digit)


def get_all_features(grammar) -> List[Feature]:
    return (
        extract_existence(grammar)
        + extract_numeric(grammar)
        + extract_length(grammar)
        + extract_is_digit(grammar)
    )


STANDARD_FEATURES = {EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE, LENGTH_FEATURE}
