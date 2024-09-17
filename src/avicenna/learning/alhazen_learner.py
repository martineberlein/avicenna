from typing import Optional, List, Iterable, Any
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_graphviz
import graphviz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from debugging_framework.input.oracle import OracleResult

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from avicenna.data import Input
from avicenna.learning.learner import CandidateLearner
from avicenna.learning.table import Candidate


class Model(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the model using the provided features and labels.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """
        Makes predictions using the trained model on the provided features.
        """
        pass

    def get_model(self) -> BaseEstimator:
        """
        Returns the underlying scikit-learn model.
        """
        return self.model


class DecisionTreeModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


class RandomForestModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


class GradientBoostingModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = GradientBoostingClassifier(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


class DataHandler:
    def __init__(self):
        self.x_data = pd.DataFrame()
        self.y_data = pd.Series(dtype='float')

    def collect_data(self, test_inputs: Iterable[Any]):
        """
        Adds features and labels from test_inputs to x_train and y_train for training a decision tree.

        Parameters:
        - test_inputs (Iterable[Input]): An iterable of Input objects containing feature data and labels.
        """
        features_list = []
        labels_list = []

        for inp in test_inputs:
            # Extract features
            features = inp.features.get_features()
            features_list.append(features)

            label: OracleResult = inp.oracle
            labels_list.append(label.is_failing())

        # Create DataFrames from the features and labels
        new_features_df = pd.DataFrame.from_records(features_list)
        new_labels_series = pd.Series(labels_list, name='label')

        # Check if the new DataFrames are not empty
        if not new_features_df.empty:
            # Initialize self.x_train and self.y_train as empty structures if they don't exist
            if not hasattr(self, 'x_train'):
                self.x_data = pd.DataFrame()
            if not hasattr(self, 'y_train'):
                self.y_data = pd.Series(dtype=new_labels_series.dtype, name='label')

            # Concatenate the new data with the existing x_train and y_train
            self.x_data = pd.concat([self.x_data, new_features_df], ignore_index=True, sort=False)
            self.y_data = pd.concat([self.y_data, new_labels_series], ignore_index=True)

    def preprocess_data(self):
        # Replace inf and -inf with NaN
        self.x_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remove columns with all NaN values
        self.x_data.dropna(axis=1, how='all', inplace=True)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        self.x_data = pd.DataFrame(imputer.fit_transform(self.x_data), columns=self.x_data.columns)

    def get_features_and_labels(self):
        return self.x_data, self.y_data


class ModelTrainer:
    def __init__(self):
        self.data_handler = DataHandler()
        self.learner = None

    def train_model(self, learner: Model, test_inputs: Iterable[Input]):
        self.data_handler.collect_data(test_inputs)
        self.data_handler.preprocess_data()
        X, y = self.data_handler.get_features_and_labels()
        learner.train(X, y)
        self.learner = learner

    def predict(self, X: pd.DataFrame):
        """
        Makes predictions using the trained model.

        Parameters:
        - X (pd.DataFrame): The feature matrix for prediction.

        Returns:
        - np.ndarray: Predicted labels.
        """
        if self.learner is None:
            raise ValueError("Model has not been trained yet.")
        return self.learner.predict(X)

    def get_model(self) -> BaseEstimator:
        """
        Returns the trained scikit-learn model.

        Returns:
        - BaseEstimator: The trained model.
        """
        if self.learner is None:
            raise ValueError("Model has not been trained yet.")
        return self.learner.get_model()

    def visualize_decision_tree(self, feature_names=None, class_names=None, figsize=(10, 10)):
        if not isinstance(self.learner, DecisionTreeModel):
            raise ValueError("Visualization is only supported for DecisionTreeLearner.")
        if self.learner.model is None:
            raise ValueError("The model has not been trained yet.")

        plt.figure(figsize=figsize)
        plot_tree(
            self.learner.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=12
        )
        plt.show()

    def visualize_decision_tree_graphviz(self, feature_names=None, class_names=None):
        if not isinstance(self.learner, DecisionTreeModel):
            raise ValueError("Visualization is only supported for DecisionTreeLearner.")
        if self.learner.model is None:
            raise ValueError("The model has not been trained yet.")

        dot_data = export_graphviz(
            self.learner.model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        graph.render("decision_tree")
        return graph

    def plot_feature_importances(self, top_n=None):
        if self.learner.model is None:
            raise ValueError("The model has not been trained yet.")

        importances = self.learner.model.feature_importances_
        feature_names = self.data_handler.x_data.columns
        indices = np.argsort(importances)[::-1]

        if top_n is not None:
            indices = indices[:top_n]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(indices)), importances[indices], color="r", align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()


class AlhazenLearner(CandidateLearner):

    def __init__(self, model: Model = None):
        super().__init__()
        self.model = model if model else DecisionTreeModel()
        self.trainer = ModelTrainer()

    def learn_candidates(self, test_inputs: Iterable[Input], **kwargs) -> Optional[BaseEstimator]:
        self.trainer.train_model(self.model, test_inputs)
        return self.trainer.get_model()

    def get_candidates(self) -> Optional[BaseEstimator]:
        return self.trainer.get_model()

    def get_best_candidates(self) -> Optional[BaseEstimator]:
        return self.trainer.get_model()