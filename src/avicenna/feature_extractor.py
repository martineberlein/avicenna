import logging
from typing import List, Set, Type, Optional, Any, Tuple
from abc import ABC, abstractmethod
import warnings

import numpy as np
from pandas import DataFrame
import shap
from fuzzingbook.Grammars import Grammar
from lightgbm import LGBMClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from avicenna.feature_collector import Feature, FeatureFactory, DEFAULT_FEATURE_TYPES
from avicenna.input import Input
from avicenna.oracle import OracleResult

# Suppress the specific SHAP warning
warnings.filterwarnings('ignore', 'LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray')


class RelevantFeatureLearner(ABC):
    def __init__(
        self,
        grammar: Grammar,
        feature_types: Optional[List[Type[Feature]]] = None,
        top_n: int = 3,
        threshold: float = 0.01,
    ):
        self.grammar = grammar
        self.features = self.construct_features(feature_types or DEFAULT_FEATURE_TYPES)
        self.top_n = top_n
        self.threshold = threshold

    def construct_features(self, feature_types: List[Type[Feature]]) -> List[Feature]:
        return FeatureFactory(self.grammar).build(feature_types)

    def learn(
        self, test_input: Set[Input]
    ) -> Tuple[Set[Feature], Set[Feature], Set[Feature]]:
        if not test_input:
            raise ValueError("Input set for learning relevant features must not be empty.")

        x_train, y_train = self.get_learning_data(test_input)
        primary_features = set(self.get_relevant_features(test_input, x_train, y_train))
        logging.info(f"Determined {primary_features} as most relevant.")
        correlated_features = self.find_correlated_features(x_train, primary_features)

        return (
            primary_features,
            correlated_features - primary_features,
            set(self.features) - primary_features.union(correlated_features),
        )

    @staticmethod
    def find_correlated_features(
        x_train: DataFrame, primary_features: Set[Feature]
    ) -> Set[Feature]:
        correlation_matrix = x_train.corr(method="spearman")

        correlated_features = {
            feature
            for primary in primary_features
            for feature, value in correlation_matrix[primary].items()
            if abs(value) > 0.7
        }
        logging.info(f"Added Features: {correlated_features} due to high correlation.")
        return correlated_features

    @abstractmethod
    def get_relevant_features(
        self, test_inputs: Set[Input], x_train: DataFrame, y_train: List[int]
    ) -> List[Feature]:
        raise NotImplementedError()

    @staticmethod
    def map_result(result: OracleResult) -> int:
        match result:
            case OracleResult.NO_BUG:
                return 0
            case OracleResult.BUG:
                return 1
            case _:
                return -1

    def get_learning_data(self, test_inputs: Set[Input]) -> Tuple[DataFrame, List[int]]:
        records = [
            {
                feature: inp.features.get_feature_value(feature)
                for feature in self.features
            }
            for inp in test_inputs
            if inp.oracle != OracleResult.UNDEF
        ]

        df = DataFrame.from_records(records).replace(-np.inf, -(2**32))
        labels = [
            self.map_result(inp.oracle)
            for inp in test_inputs
            if inp.oracle != OracleResult.UNDEF
        ]

        return df.drop(columns=df.columns[df.nunique() == 1]), labels


class SKLearFeatureRelevanceLearner(RelevantFeatureLearner, ABC):
    def get_features(self, x_train: DataFrame, classifier) -> List[Feature]:
        features_with_importance = list(
            zip(x_train.columns, classifier.feature_importances_)
        )

        sorted_features = sorted(
            features_with_importance, key=lambda x: x[1], reverse=True
        )
        important_features = [
            feature
            for feature, importance in sorted_features
            if importance >= self.threshold
        ][: self.top_n]

        return important_features

    @abstractmethod
    def fit(self, x_train: DataFrame, y_train: List[int]) -> Any:
        raise NotImplementedError()

    def get_relevant_features(
        self, test_inputs: Set[Input], x_train: DataFrame, y_train: List[int]
    ) -> List[Feature]:
        classifier = self.fit(x_train, y_train)
        return self.get_features(x_train, classifier)


class DecisionTreeRelevanceLearner(SKLearFeatureRelevanceLearner):
    def fit(self, x_train: DataFrame, y_train: List[int]) -> Any:
        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(x_train, y_train)
        return classifier


class RandomForestRelevanceLearner(SKLearFeatureRelevanceLearner):
    def fit(self, x_train: DataFrame, y_train: List[int]) -> Any:
        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        classifier.fit(x_train, y_train)
        return classifier


class GradientBoostingTreeRelevanceLearner(SKLearFeatureRelevanceLearner):
    def fit(self, x_train: DataFrame, y_train: List[int]) -> Any:
        classifier = LGBMClassifier(max_depth=5, n_estimators=3000, objective="binary")
        classifier.fit(x_train, y_train)
        return classifier


class SHAPRelevanceLearner(RelevantFeatureLearner):
    def __init__(
        self,
        grammar: Grammar,
        top_n: int = 3,
        feature_types: Optional[List[Type[Feature]]] = None,
        classifier_type: Optional[
            Type[SKLearFeatureRelevanceLearner]
        ] = GradientBoostingTreeRelevanceLearner,
        normalize_data: bool = False,
        show_beeswarm_plot: bool = False,
    ):
        super().__init__(grammar, top_n=top_n, feature_types=feature_types)
        self.classifier = classifier_type(self.grammar)
        self.show_beeswarm_plot = show_beeswarm_plot
        self.normalize_data = normalize_data

    def get_relevant_features(
        self, test_inputs: Set[Input], x_train: DataFrame, y_train: List[int]
    ) -> List[Feature]:
        x_train_normalized = self.normalize_learning_data(x_train)
        classifier = self.classifier.fit(x_train_normalized, y_train)
        shap_values = self.get_shap_values(classifier, x_train)
        if self.show_beeswarm_plot:
            self.display_beeswarm_plot(shap_values, x_train)
        return self.get_sorted_features_by_importance(shap_values, x_train)[
            : self.top_n
        ]

    def normalize_learning_data(self, data: DataFrame):
        if self.normalize_data:
            normalized = preprocessing.MinMaxScaler().fit_transform(data)
            return DataFrame(normalized, columns=data.columns)
        else:
            return data

    @staticmethod
    def get_shap_values(classifier, x_train):
        explainer = shap.TreeExplainer(classifier)
        return explainer.shap_values(x_train)

    @staticmethod
    def get_sorted_features_by_importance(
        shap_values, x_train: DataFrame
    ) -> List[Feature]:
        mean_shap_values = np.abs(shap_values[1]).mean(axis=0)
        sorted_indices = mean_shap_values.argsort()[::-1]
        return x_train.columns[sorted_indices].tolist()

    @staticmethod
    def display_beeswarm_plot(shap_values, x_train):
        return shap.summary_plot(shap_values[1], x_train.astype("float"))
