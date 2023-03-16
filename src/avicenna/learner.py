import logging
from typing import List, Tuple, Set
from fuzzingbook.Grammars import Grammar
from fuzzingbook.Parser import EarleyParser
from grammar_graph.gg import GrammarGraph
from isla.language import DerivationTree

from avicenna.fuzzer.generator import Generator
from avicenna.feature_collector import Collector
from avicenna.feature_extractor import Extractor, get_all_non_terminals
from avicenna.features import STANDARD_FEATURES, FeatureWrapper, Feature


class InputElementLearner:
    """
    The learner class that determines the most relevant input elements for failure-inducing inputs according to a
    given grammar. Main method: learn().
    """

    def __init__(
        self,
        grammar: Grammar,
        prop,
        input_samples: List[DerivationTree],
        generate_more_inputs: bool = True,
        max_relevant_features: int = 2,
        features: Set[FeatureWrapper] = STANDARD_FEATURES,
        show_shap_beeswarm=False,
    ):
        """
        Constructs a new learner object. Parsing a grammar, a property-function and a set of initial inputs is
        mandatory.
        """
        self._grammar = grammar
        self._prop = prop
        self._inputs = input_samples
        self._generate_more_inputs: bool = generate_more_inputs
        self._max_positive_samples: int = 50
        self._max_negative_samples: int = 50
        self._max_relevant_features: int = max_relevant_features
        self._collector = None
        self._extractor = None
        self._features = features
        self._show_shap_beeswarm = show_shap_beeswarm
        self._relevant_input_elements = None

    def learn(self, use_correlation=True) -> List[Tuple[str, Feature, List[Feature]]]:
        """
        Learns and determines the most relevant input elements by (1) parsing the input files into their grammatical
        features; (2) training a machine learning model that associates the input features with the failure/passing
        of the inputs and (3) extracting the decisions made by the machine learning model with the ML-Explainer SHAP.

        :return: A list of tuples, containing the most relevant non-terminals and the corresponding feature.
        """

        assert (
            self._inputs is not None
        ), "Learner needs at least one failure inducing input."

        exec_oracle = []
        for inp in self._inputs:
            exec_oracle.append(self._prop(inp))

        positive_samples = []
        negative_samples = []

        if self._inputs is not None:
            for i, inp in enumerate(self._inputs):
                if exec_oracle[i] is True:
                    positive_samples.append(inp)
                else:
                    negative_samples.append(inp)

            logging.info(
                f"Starting with {len(positive_samples)} failure-inducing and {len(negative_samples)} benign "
                f"inputs."
            )
        else:
            #  TODO Use Grammar Fuzzer
            pass

        if all(isinstance(inp, DerivationTree) for inp in self._inputs):
            pass
        else:
            positive_samples = [
                DerivationTree.from_parse_tree(
                    next(EarleyParser(self._grammar).parse(inp))
                )
                for inp in positive_samples
            ]

        if self._generate_more_inputs or self._inputs is None:
            logging.info(f"Generating more inputs.")
            generator = Generator(
                self._max_positive_samples,
                self._max_negative_samples,
                self._grammar,
                self._prop,
            )
            pos_inputs, neg_inputs = generator.generate_mutation(positive_samples)
            positive_samples = pos_inputs
            negative_samples = neg_inputs

        logging.info(
            f"Learning with {len(positive_samples)} failure-inducing and {len(negative_samples)} benign "
            f"inputs."
        )
        input_list = positive_samples + negative_samples
        self._collector = Collector(self._grammar, self._features)
        logging.info(f"Collecting and parsing features.")

        feature_table = self._collector.collect_features(input_list)
        oracle_table = [self._prop(inp) for inp in input_list]

        combined_data = feature_table.copy()
        combined_data["oracle"] = oracle_table

        logging.info(f"Learning most relevant input elements.")
        self._extractor = Extractor(
            combined_data,
            self._grammar,
            max_feature_num=self._max_relevant_features,
            features=self._features,
        )
        ex = self._extractor.extract_non_terminals()

        if self._show_shap_beeswarm:
            self.show_beeswarm_plot()

        final_results = self._extractor.get_clean_most_important_features()
        relevant_input_elements = []
        for feature in self._extractor.get_clean_most_important_features():
            corr_features: List[Feature] = self._extractor.get_correlating_features(
                feature
            )
            relevant_input_elements.append((str(feature.rule), feature, corr_features))
        logging.info(f"Extracted {len(relevant_input_elements)} {[f[0] for f in relevant_input_elements]} input "
                     f"elements.")

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
