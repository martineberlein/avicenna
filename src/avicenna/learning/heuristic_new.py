import logging
from typing import List, Optional, Iterable, Set

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from isla.language import Formula
from isla.language import ISLaUnparser
from grammar_graph import gg

from debugging_framework.input.oracle import OracleResult
from debugging_framework.fuzzingbook.grammar import Grammar

from avicenna.learning.learner import TruthTablePatternCandidateLearner
from avicenna.learning.table import Candidate, CandidateSet
from avicenna.data import Input

logger = logging.getLogger("learner")


# Model Classes
class Model:
    def __init__(self):
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame):
        pass

    def get_model(self) -> BaseEstimator:
        return self.model


class DecisionTreeModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        remove_unequal_decisions(self.model)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


# Data Handler Class
class CandidateDataHandler:
    def __init__(self):
        self.x_data = pd.DataFrame()
        self.y_data = pd.Series(dtype='bool')
        self.candidate_names = []
        self.test_input_labels = {}  # Mapping from test input identifier to label

    def collect_data(self, candidates: CandidateSet, positive_inputs: Set, negative_inputs: Set):
        """
        Constructs the DataFrame from the candidates by directly accessing their evaluation results.
        The columns are the candidate formulas, and the rows are the test inputs.
        """
        # Combine all inputs
        all_test_inputs = list(positive_inputs) + list(negative_inputs)
        # Generate unique identifiers for test inputs (ensure consistent ordering)
        test_input_ids = [str(inp) for inp in all_test_inputs]

        # Create the label Series y_data
        labels = [True] * len(positive_inputs) + [False] * len(negative_inputs)
        self.y_data = pd.Series(labels, index=test_input_ids, dtype='bool')

        # Initialize an empty DataFrame with index as test_input_ids
        self.x_data = pd.DataFrame(index=test_input_ids)

        # For each candidate, create a column in x_data
        for candidate in candidates:
            candidate_name = str(candidate.formula)
            candidate_name = ISLaUnparser(candidate.formula).unparse()
            self.candidate_names.append(ISLaUnparser(candidate.formula).unparse())

            # Initialize a list to hold evaluation results for all inputs
            eval_results = []

            # Access candidate.comb[inp] for each input
            for inp in all_test_inputs:
                eval_result = candidate.comb.get(inp, np.nan)
                eval_results.append(eval_result)

            print(candidate_name)
            print(eval_results)

            # Add the evaluation results as a column in x_data
            self.x_data[candidate_name] = eval_results

    def preprocess_data(self):
        # Fill missing values with False (assuming Boolean features)
        self.x_data = self.x_data.fillna(False)

    def get_features_and_labels(self):
        return self.x_data, self.y_data


# Model Trainer Class
class ModelTrainer:
    def __init__(self):
        self.learner = None

    def train_model(self, model: Model, X: pd.DataFrame, y: pd.Series):
        model.train(X, y)
        self.learner = model

    def predict(self, X: pd.DataFrame):
        if self.learner is None:
            raise ValueError("Model has not been trained yet.")
        return self.learner.predict(X)

    def get_model(self) -> BaseEstimator:
        if self.learner is None:
            raise ValueError("Model has not been trained yet.")
        return self.learner.get_model()

    def visualize_decision_tree(self, feature_names=None, class_names=None, figsize=(10, 10)):
        if not isinstance(self.learner, DecisionTreeModel):
            raise ValueError("Visualization is only supported for DecisionTreeModel.")
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


# Refactored HeuristicTreePatternCandidateLearner
class HeuristicTreePatternCandidateLearner(TruthTablePatternCandidateLearner):
    def __init__(self, grammar: Grammar) -> None:
        super().__init__(grammar)
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.model_trainer = ModelTrainer()
        self.data_handler = CandidateDataHandler()

    def learn_candidates(
        self,
        test_inputs: Set,
        exclude_nonterminals: Optional[Iterable[str]] = None,
    ) -> Optional[List[Candidate]]:
        # Categorize inputs
        positive_inputs, negative_inputs = self.categorize_inputs(test_inputs)
        self.update_inputs(positive_inputs, negative_inputs)

        # Construct atomic candidates
        atomic_formulas = self.atomic_candidate_constructor.construct_candidates(
            positive_inputs=self.all_positive_inputs,
            exclude_nonterminals=exclude_nonterminals
        )
        new_atomic_candidates = {Candidate(formula) for formula in atomic_formulas}

        # Update candidate set with new atomic candidates
        for candidate in new_atomic_candidates:
            if candidate not in self.candidates:
                self.candidates.append(candidate)

        # Evaluate the candidates
        for candidate in self.candidates:
            self.evaluate_formula(candidate, positive_inputs, negative_inputs)

        candidates_to_remove = []
        for candidate in self.candidates:
            if candidate.recall() < self.min_recall:
                candidates_to_remove.append(candidate)
        for cand in candidates_to_remove:
            self.candidates.remove(cand)

        # Collect data using DataHandler
        self.data_handler.collect_data(self.candidates, self.all_positive_inputs, self.all_negative_inputs)
        self.data_handler.preprocess_data()
        X, y = self.data_handler.get_features_and_labels()

        n_inputs = len(self.all_positive_inputs) + len(self.all_negative_inputs)
        min_samples_split = max(2, int(0.05 * n_inputs))  # 5% of the total samples or at least 2
        min_samples_leaf = max(1, int(0.01 * n_inputs))  # 1% of the total samples or at least 1

        # Choose a model
        model = DecisionTreeModel(
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=3,
            max_depth=3,
            class_weight="balanced",
        )

        # Train the model using ModelTrainer
        self.model_trainer.train_model(model, X, y)

        # Visualize the decision tree (optional)
        # self.model_trainer.visualize_decision_tree(feature_names=self.data_handler.candidate_names)

        # Optionally extract rules from the decision tree to generate new candidates

        # Return the trained model or the generated candidates
        # return self.get_best_candidates()

    def evaluate_formula(
        self,
        candidate: Candidate,
        positive_inputs: Set[Input],
        negative_inputs: Set[Input],
    ):
        """
        Evaluates a formula on a set of inputs.
        """
        if candidate in self.candidates:
            candidate.evaluate(positive_inputs, self.graph)
            if candidate.recall() < self.min_recall:
                # filter out candidates that do not meet the recall threshold
                # self.candidates.remove(candidate)
                pass
            else:
                # evaluate the candidate on the remaining negative inputs
                candidate.evaluate(negative_inputs, self.graph)
        else:
            candidate.evaluate(self.all_positive_inputs, self.graph)
            candidate.evaluate(self.all_negative_inputs, self.graph)
            if candidate.recall() < self.min_recall:
                # self.candidates.remove(candidate)
                pass

    def get_best_candidates(self) -> Optional[List[Candidate]]:
        """
        Extracts the best candidates based on the trained model.
        """
        # Implement logic to extract best candidates from the model
        # This could involve interpreting the decision tree paths
        # For simplicity, returning all candidates with high recall
        return [candidate for candidate in self.candidates if candidate.recall() >= self.min_recall]

    def extract_positive_rules(self, model: DecisionTreeClassifier, feature_names: List[str]) -> List[List[str]]:
        """
        Extracts rules from the decision tree that lead to positive predictions.

        Parameters:
        - model: Trained DecisionTreeClassifier.
        - feature_names: List of feature names.

        Returns:
        - List of rules, where each rule is a list of conditions.
        """
        from sklearn.tree import _tree

        tree_ = model.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Decision node
                name = feature_name[node]
                threshold = tree_.threshold[node]
                # Left child
                if threshold <= 0.5:
                    # Feature value is False (0)
                    left_conditions = conditions.copy()
                    left_conditions.append(f"NOT {name}")
                    recurse(tree_.children_left[node], left_conditions)
                    # Right child
                    right_conditions = conditions.copy()
                    right_conditions.append(f"{name}")
                    recurse(tree_.children_right[node], right_conditions)
                else:
                    # Since features are Boolean, thresholds should be at 0.5
                    pass  # Handle accordingly if needed
            else:
                # Leaf node
                value = tree_.value[node]
                class_idx = np.argmax(value)
                # Assuming binary classification with classes [0,1]
                if class_idx == 1:
                    # Positive prediction
                    rules.append(conditions)

        recurse(0, [])
        return rules


def is_leaf(clf, node: int) -> bool:
    """returns true if the given node is a leaf."""
    return clf.tree_.children_left[node] == clf.tree_.children_right[node]


def leaf_label(clf, node: int) -> int:
    """returns the index of the class at this node. The node must be a leaf."""
    assert is_leaf(clf, node)
    occs = clf.tree_.value[node][0]
    idx = 0
    maxi = occs[idx]
    for i, o in zip(range(0, len(occs)), occs):
        if maxi < o:
            maxi = o
            idx = i
    return idx



def remove_unequal_decisions(clf):
    """
    This method rewrites a decision tree classifier to remove nodes where the same
    decision is taken on both sides.

    :param clf: a decision tree classifier
    :return: the same classifier, rewritten
    """
    changed = True
    while changed:
        changed = False
        for node in range(0, clf.tree_.node_count):
            if not is_leaf(clf, node) and (
                    is_leaf(clf, clf.tree_.children_left[node])
                    and is_leaf(clf, clf.tree_.children_right[node])
            ):
                # both children of this node are leaves
                left_label = leaf_label(clf, clf.tree_.children_left[node])
                right_label = leaf_label(clf, clf.tree_.children_right[node])
                if left_label == right_label:
                    clf.tree_.children_left[node] = -1
                    clf.tree_.children_right[node] = -1
                    clf.tree_.feature[node] = -2
                    changed = True
                    assert left_label == leaf_label(clf, node)
    return clf