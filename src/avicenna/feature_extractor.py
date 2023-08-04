import logging
import warnings
from typing import List, Set

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import shap
from fuzzingbook.Grammars import Grammar, nonterminals
from lightgbm import LGBMClassifier
from sklearn import preprocessing

from avicenna.features import Feature, FeatureWrapper, STANDARD_FEATURES
from avicenna.feature_collector import get_all_features_n


CORRELATION_THRESHOLD = 0.7


def get_all_non_terminals(grammar):
    non_terminals = [n for n in grammar]
    return non_terminals


class Extractor:
    def __init__(
        self, complete_data: DataFrame, grammar, max_feature_num: int = 4, features=None
    ):
        self._data: DataFrame = complete_data
        self._X_train = complete_data.drop(["oracle"], axis=1)
        self._y_train = complete_data["oracle"].astype(str)
        self._grammar: Grammar = grammar
        if features is None:
            features = STANDARD_FEATURES
        self._features: Set[FeatureWrapper] = features
        self._all_features = get_all_features_n(features, grammar)
        self._clf = None
        self._shap_values = None
        self._most_important_features = None
        self._exclusion_set = None
        self._max_num = max_feature_num
        self._classifier = (
            None  # TODO allow to specify different machine learning models
        )
        self._correlation_matrix = None

    def get_most_important_features(self):
        return self._most_important_features

    def get_clean_most_important_features(self):
        clean_features = []
        for i in self._most_important_features:
            # Stop if nummer of max features is reached
            if len(clean_features) >= self._max_num:
                break
            # Only consider those existence features that have a positive influence on the outcome
            # print(self._data['exists(<maybe_minus>@1)'])
            for j in self._all_features:
                if i == j.name:
                    clean_features.append(j)

        return clean_features

    def get_correlating_features(self, feature: Feature) -> List[Feature]:
        assert self._correlation_matrix is not None, "No correlation matrix available."

        corr_features: List[Feature] = []
        for i in self._correlation_matrix.columns:
            corr_feature = self._get_feature(i)

            corr_value = self._correlation_matrix[feature.__repr__()][i]
            if corr_value > CORRELATION_THRESHOLD and feature.__repr__() != i:
                if corr_feature is not None:
                    corr_features.append(corr_feature)
                else:
                    logging.info(
                        f"Feature {feature} also correlates with {i}, which is not a feature."
                    )
                    # TODO is this a good idea?
                    corr_features.append(feature)
        return corr_features

    def extract_non_terminals(self) -> List[str]:
        self._normalize_data()
        self._calculate_correlation_matrix()
        self._learn_classifier()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._explain_prediction()
        self._get_exclusion_set()

        return self._exclusion_set

    def _learn_classifier(self):
        self._clf = LGBMClassifier(max_depth=5, n_estimators=3000, objective="binary")
        self._clf.fit(self._X_train, self._y_train)

    def _explain_prediction(self):
        explainer = shap.TreeExplainer(self._clf)
        self._shap_values = explainer.shap_values(self._X_train)
        sv = np.array(self._shap_values)
        y_bool = self._clf.predict(self._X_train).astype(bool)

    def _get_exclusion_set(self):
        _features = self._X_train.columns

        idx = np.abs(np.array(self._shap_values[0])).mean(0).argsort()
        # Two most important non-terminals that are responsible for the failure (according to the classifier)
        self._most_important_features = list(_features[idx[::-1]])

        excluded_features = set(get_all_non_terminals(self._grammar))
        clean_features = self.get_clean_most_important_features()

        for i in clean_features:
            if i.rule in excluded_features:
                excluded_features.remove(i.rule)
                for child in nonterminals(i.key):
                    if child in excluded_features:
                        excluded_features.remove(child)

        self._exclusion_set = excluded_features

    def show_beeswarm_plot(self):
        return shap.summary_plot(self._shap_values[1], self._X_train.astype("float"))
        # return shap.summary_plot(self._shap_values)

    def _normalize_data(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        np_array = self._data.to_numpy()
        normalized = min_max_scaler.fit_transform(np_array)
        new_data = DataFrame(normalized, columns=self._data.columns)
        # self._data = new_data
        self._X_train = new_data.drop("oracle", axis=1)
        self._y_train = new_data["oracle"]

    def _calculate_correlation_matrix(self):
        corr_data: DataFrame = self._data

        for col in corr_data.columns:
            if (corr_data[col].values == 1.0).all():
                # print("Dropping: ", col)
                # corr_data = corr_data.drop(col, axis=1)
                pass

        corr_matrix = corr_data.corr(method="spearman")
        corr = []
        for i in corr_matrix.columns:
            for j in corr_matrix.columns:
                if abs(corr_matrix[i][j]) > CORRELATION_THRESHOLD and i != j:
                    corr.append((i, j, corr_matrix[i][j]))
        self._correlation_matrix = corr_matrix

    def _get_feature(self, feature_name: str) -> Feature:
        for feature in self._all_features:
            if feature.__repr__() == feature_name:
                return feature


def show_correlation_heatmap(data, corr_matrix):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(
        range(data.select_dtypes(["number"]).shape[1]),
        data.select_dtypes(["number"]).columns,
        fontsize=14,
        rotation=45,
    )
    plt.yticks(
        range(data.select_dtypes(["number"]).shape[1]),
        data.select_dtypes(["number"]).columns,
        fontsize=14,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()
