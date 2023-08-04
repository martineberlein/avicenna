import logging
import sys
from typing import List, Tuple, Set, Callable
from pandas import DataFrame
from fuzzingbook.Grammars import Grammar
from fuzzingbook.Parser import EarleyParser
from grammar_graph.gg import GrammarGraph
from isla.language import DerivationTree

from avicenna.generator import Generator
from avicenna.feature_collector import Collector
from avicenna.feature_extractor import Extractor, get_all_non_terminals
from avicenna.features import STANDARD_FEATURES, FeatureWrapper, Feature
from avicenna.input import Input
from avicenna.oracle import OracleResult


class InputElementLearner:
    """
    The learner class that determines the most relevant input elements for failure-inducing inputs according to a
    given grammar. Main method: learn().
    """

    def __init__(
        self,
        grammar: Grammar,
        oracle,
        max_relevant_features: int = 2,
        generate_more_inputs: bool = True,
        features: Set[FeatureWrapper] = STANDARD_FEATURES,
        show_shap_beeswarm=False,
    ):
        """
        Constructs a new learner object. Parsing a grammar, a property-function and a set of initial inputs is
        mandatory.
        """
        self._grammar = grammar
        self._prop: Callable[[DerivationTree], bool] = oracle
        self._generate_more_inputs: bool = generate_more_inputs
        self._max_positive_samples: int = 100
        self._max_negative_samples: int = 100
        self._max_relevant_features: int = max_relevant_features
        self._collector = None
        self._extractor = None
        self._features = features
        self._show_shap_beeswarm = show_shap_beeswarm
        self._relevant_input_elements = None

    def learn(
        self, test_inputs: Set[Input], use_correlation=True
    ) -> List[Tuple[str, Feature, List[Feature]]]:
        """
        Learns and determines the most relevant input elements by (1) parsing the input files into their grammatical
        features; (2) training a machine learning model that associates the input features with the failure/passing
        of the inputs and (3) extracting the decisions made by the machine learning model with the ML-Explainer SHAP.

        :return: A list of tuples, containing the most relevant non-terminals and the corresponding feature.
        """

        assert (
            test_inputs is not None
        ), "Learner needs at least one failure inducing input."

        if not all(map(lambda x: isinstance(x.oracle, OracleResult), test_inputs)):
            for inp in test_inputs:
                label = self._prop(inp.tree)
                inp.oracle = OracleResult.BUG if label else OracleResult.NO_BUG

        num_bug_inputs = len(
            [inp for inp in test_inputs if inp.oracle == OracleResult.BUG]
        )

        logging.info(
            f"Learning with {num_bug_inputs} failure-inducing and {len(test_inputs) - num_bug_inputs} benign "
            f"inputs."
        )

        logging.info(f"Collecting and parsing features.")
        self._collector = Collector(self._grammar, self._features)

        for inp in test_inputs:
            if inp.features is None:
                inp.features = self._collector.collect_features(inp)

        learning_data = self._get_learning_data(test_inputs)

        logging.info(f"Learning most relevant input elements.")
        self._extractor = Extractor(
            learning_data,
            self._grammar,
            max_feature_num=self._max_relevant_features,
            features=self._features,
        )
        ex = self._extractor.extract_non_terminals()
        final_results = self._extractor.get_clean_most_important_features()

        if self._show_shap_beeswarm:
            self.show_beeswarm_plot()

        relevant_input_elements = []
        for feature in self._extractor.get_clean_most_important_features():
            corr_features: List[Feature] = self._extractor.get_correlating_features(
                feature
            )
            relevant_input_elements.append((str(feature.rule), feature, corr_features))
        logging.info(
            f"Extracted {len(relevant_input_elements)} {[f[0] for f in relevant_input_elements]} input "
            f"elements."
        )

        if use_correlation:
            prev_length = len(relevant_input_elements)
            relevant_input_elements = (
                self._determine_best_correlation_parent_non_terminal(
                    relevant_input_elements
                )
            )
            logging.info(
                f"Added {len(relevant_input_elements) - prev_length} additional input element(s) with high "
                f"correlation."
            )

        for elem in relevant_input_elements:
            logging.debug(elem)

        self._relevant_input_elements = relevant_input_elements
        return relevant_input_elements

    def show_beeswarm_plot(self):
        return self._extractor.show_beeswarm_plot()

    def get_exclusion_sets(self, use_correlation=True) -> Tuple[Set[str], Set[str]]:
        relevant_input_elements = self._relevant_input_elements
        if relevant_input_elements is None:
            relevant_input_elements = self.learn(use_correlation=use_correlation)

        non_terminals = set([non_term[0] for non_term in relevant_input_elements])
        all_features = set(get_all_non_terminals(self._grammar))

        return non_terminals, all_features - non_terminals

    def _determine_best_correlation_parent_non_terminal(
        self, relevant_input_elements: List[Tuple[str, Feature, List[Feature]]]
    ):
        """
        If len(<string>) is relevant and len(<payload>) is in the set of highly correlating features, we
        want len(<payload>) to be returned as this is the more general non_terminal.
        Heuristic: Select that non-terminal from the set of correlating features that is the highest up in the
        Derivation Tree/Grammar.
        """
        graph = GrammarGraph.from_grammar(self._grammar)

        extended_relevant_input_elements = relevant_input_elements

        for relevant_input_element in relevant_input_elements:
            non_terminal, feature, corr_features = relevant_input_element
            for corr_feature in corr_features:
                if (
                    graph.reachable(corr_feature.rule, feature.rule)
                    and not (graph.reachable(feature.rule, corr_feature.rule))
                    and not corr_feature.rule == "<start>"
                ):
                    extended_relevant_input_elements.append(
                        (
                            str(corr_feature.rule),
                            corr_feature,
                            self._extractor.get_correlating_features(corr_feature),
                        )
                    )

        return extended_relevant_input_elements

    @staticmethod
    def _get_learning_data(test_inputs: Set[Input]) -> DataFrame:
        data = []
        for inp in test_inputs:
            if inp.oracle != OracleResult.UNDEF:
                learning_data = inp.features  # .drop(["sample"], axis=1)
                learning_data["oracle"] = (
                    True if inp.oracle == OracleResult.BUG else False
                )
                data.append(learning_data)

        return DataFrame.from_records(data)
