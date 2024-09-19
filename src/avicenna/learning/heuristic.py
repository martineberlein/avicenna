import logging
from typing import List, Tuple, Dict, Optional, Iterable, Sequence, Set

from sklearn.tree import DecisionTreeClassifier
from isla.language import ISLaUnparser
from avicenna.learning.treetools import (
    all_path,
    prediction_for_path,
)

import pandas
import numpy
from isla.language import Formula, ConjunctiveFormula
from grammar_graph import gg

from debugging_framework.input.oracle import OracleResult
from debugging_framework.fuzzingbook.grammar import Grammar
from ..data import Input
from .table import Candidate, CandidateSet
from .learner import TruthTablePatternCandidateLearner, PatternCandidateLearner

logger = logging.getLogger("learner")

"""
>>>> Idea <<<<
- Rule Induction (Decision Tree) subclass of TruthTablePatternCandidateLearner
- Use the Decision Tree to learn Decision Rules
- Use the Decision Rules to learn the Invariants
- Other Rule induction algorithms can be used as well (e.g. RuleFit, Random Forest, etc.)

>>>> Implementation <<<<
"""


class HeuristicTreePatternCandidateLearner(TruthTablePatternCandidateLearner):

    def __init__(self, grammar: Grammar) -> None:
        super().__init__(grammar)
        self.graph = gg.GrammarGraph.from_grammar(grammar)

    def learn_candidates(self, test_inputs: Set[Input], exclude_nonterminals: Optional[Iterable[str]] = None,) -> Optional[List[Candidate]]:
        # get all atomic candidates
        # construct data frame for decision-tree-learning.
        # learn the decision tree on the atomic candidates
        # extract the rules from the decision tree
        # learn the invariants from the rules
        positive_inputs, negative_inputs = self.categorize_inputs(test_inputs)
        self.update_inputs(positive_inputs, negative_inputs)
        atomic_formulas = self.atomic_candidate_constructor.construct_candidates(
            positive_inputs=self.all_positive_inputs,
            exclude_nonterminals=exclude_nonterminals
        )
        new_atomic_candidates = {Candidate(formula) for formula in atomic_formulas}

        # Get all previously learned candidates + the new atomic candidates
        tmp_candidates = set(self.candidates.candidates)
        for candidate in new_atomic_candidates:
            if candidate not in tmp_candidates:
                tmp_candidates.add(candidate)

        # Evaluate the atomic candidates
        for candidate in tmp_candidates:
            self.evaluate_formula(candidate, positive_inputs, negative_inputs)

        # Construct the data frame for the decision tree learning
        dataframe = self.construct_dataframe(self.candidates)
        cls = self._learn_decision_tree(dataframe)

        return dataframe

        # remove if not successful

    def construct_dataframe(self, candidates: CandidateSet) -> pandas.DataFrame:
        """
        Construct a DataFrame from the candidates for the decision tree learning.
        The columns are the formula evaluations for the failing and passing inputs.
        Formulas are represented by their indices in the candidate set.
        """
        result = pandas.DataFrame()
        length_inputs = len(self.all_positive_inputs) + len(self.all_negative_inputs)

        for candidate in candidates:
            result[candidate] = pandas.Series([numpy.nan] * length_inputs)
            result[candidate] = candidate.failing_inputs_eval_results + candidate.passing_inputs_eval_results

        result["oracle"] = [True] * len(self.all_positive_inputs) + [False] * len(self.all_negative_inputs)
        # result["test_input"] = [inp for inp in self.all_positive_inputs] + [inp for inp in self.all_negative_inputs]

        return result

    @staticmethod
    def _learn_decision_tree(dataframe: pandas.DataFrame) -> DecisionTreeClassifier:
        """
        Learn a decision tree from the given data frame.
        """
        clf = DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=2,  # minimal value
            max_features=5,
            max_depth=3
        )
        clf = clf.fit(dataframe.drop("oracle", axis=1), dataframe["oracle"])
        return clf

    def get_candidates(self) -> Optional[List[Candidate]]:
        pass

    def get_best_candidates(self) -> Optional[List[Candidate]]:
        pass

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
                self.candidates.remove(candidate)
            else:
                # evaluate the candidate on the remaining negative inputs
                candidate.evaluate(negative_inputs, self.graph)
        else:
            candidate.evaluate(self.all_positive_inputs, self.graph)
            candidate.evaluate(self.all_negative_inputs, self.graph)
            if candidate.recall() >= self.min_recall:
                self.candidates.append(candidate)

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


class Safe():

    def learn_candidates(self, test_inputs: Iterable[Input], **kwargs) -> Optional[List[Candidate]]:
        pass

    def evaluate_candidates(self, candidates: Set[Candidate]):
        pass

    def _learn_invariants(
        self,
        positive_inputs: Set[Input],
        negative_inputs: Set[Input],
    ):
        self.all_positive_inputs = positive_inputs  # Todo not correct
        self.all_negative_inputs = negative_inputs
        # sorted_positive_inputs = self.sort_and_filter_inputs(self.all_positive_inputs)
        # candidates = self.get_candidates(sorted_positive_inputs)
        # Todo: Implement the following method
        #   candidates = self.get_candidates(positive_inputs)
        #   get the atomic Candidates from Super Class
        candidates: Set[Formula] = set()
        print(f"Number of candidates: ", len(candidates))

        new_candidates = {Candidate(formula) for formula in candidates}
        # Todo: Use all Positive Inputs and Negative Inputs to evaluate the Candidate Formula
        self.evaluate_candidates(new_candidates)

        dataframe = self.build_dataframe(new_candidates)
        candidates = self.learn_decision_tree(dataframe)

        return candidates

    def _build_dataframe(self, candidate_set: CandidateSet) -> pandas.DataFrame:
        """
        Build a DataFrame from the candidate set for the decision tree learning.
        The columns are the formula evaluations for the failing and passing inputs.
        Formulas are represented by their indices in the candidate set.
        """
        result = pandas.DataFrame()
        length_inputs = len(self.all_positive_inputs) + len(self.all_negative_inputs)

        for idx, candidate in enumerate(candidate_set):
            result[idx] = pandas.Series([numpy.nan] * length_inputs)
            result[idx] = candidate.failing_inputs_eval_results + candidate.passing_inputs_eval_results

        result["oracle"] = [True] * len(self.all_positive_inputs) + [False] * len(self.all_negative_inputs)

        return result

    @staticmethod
    def _learn_decision_tree(dataframe: pandas.DataFrame) -> DecisionTreeClassifier:
        """
        Learn a decision tree from the given data frame.
        """
        clf = DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=2,  # minimal value
            max_features=5,
            max_depth=3,
            class_weight={True: 1.0, False: 1.0},
        )
        clf = clf.fit(dataframe.drop("oracle", axis=1), dataframe["oracle"])
        return clf

    def extract_rules(self, clf: DecisionTreeClassifier, candidate_set: CandidateSet, feature_names: List[str]) -> Set[Candidate]:
        """
        Extract the rules from the decision tree classifier.
        """
        all_paths = all_path(clf)
        candidates: Set[Candidate] = set()
        # Todo: what about negations? (e.g. not a and b)
        for path in all_paths:
            if prediction_for_path(clf, path) == OracleResult.FAILING:
                candidate = candidate_set[clf.tree_.feature[path[0]]]
                for elem in path[1:]:
                    candidate = candidate.__and__(candidate_set[clf.tree_.feature[elem]])

                candidates.add(candidate)

        return candidates

    # def learn_decision_tree(
    #     self, dataframe: pandas.DataFrame
    # ) -> List[Tuple[Formula, None, None]]:
    #
    #     sample_bug_count = len(dataframe[(dataframe["oracle"] == True)])
    #     print("number of bugs found: ", sample_bug_count)
    #     sample_count = len(dataframe)
    #     print(f"Learning with {sample_bug_count} failure inputs of {sample_count}")
    #
    #     clf = DecisionTreeClassifier(
    #         max_features=5,
    #         max_depth=3,
    #     )
    #     clf = clf.fit(dataframe.drop("oracle", axis=1), dataframe["oracle"])
    #
    #
    #     names = [
    #         ISLaUnparser(self.precision_truth_table.rows[idx].formula).unparse()
    #         for idx in dataframe.drop("oracle", axis=1).columns
    #     ]
    #
    #     # print(grouped_rules(clf, feature_names=names))
    #     # print(names)
    #     tree_text = export_text(clf, feature_names=names)
    #     # print(tree_text)
    #
    #     paths = all_path(clf)
    #
    #     candidates: List[Tuple[Formula, None, None]] = []
    #     for path in paths:
    #         print(path, prediction_for_path(clf, path))
    #         if prediction_for_path(clf, path) == OracleResult.FAILING:
    #             precision_row = self.precision_truth_table.rows[
    #                 clf.tree_.feature[path[0]]
    #             ]
    #             recall_row = self.recall_truth_table.rows[clf.tree_.feature[path[0]]]
    #
    #             for elem in path[1:-1]:
    #                 new_precision_row = self.precision_truth_table.rows[
    #                     clf.tree_.feature[elem]
    #                 ]
    #                 precision_row = precision_row.__and__(new_precision_row)
    #
    #                 new_recall_row = self.recall_truth_table.rows[
    #                     clf.tree_.feature[elem]
    #                 ]
    #                 recall_row = recall_row.__and__(new_recall_row)
    #
    #             self.precision_truth_table.append(precision_row)
    #             self.recall_truth_table.append(recall_row)
    #             t = (precision_row.formula, None, None)
    #             # form: Formula = self.precision_truth_table.rows[
    #             #     clf.tree_.feature[path[0]]
    #             # ].formula
    #             # assert isinstance(form, Formula)
    #             # for elem in path[1:-1]:
    #             #     new = self.precision_truth_table.rows[
    #             #         clf.tree_.feature[elem]
    #             #     ].formula
    #             #     assert isinstance(new, Formula)
    #             #     form = form.__and__(new)
    #             #     # print("Elem: ", names[clf.tree_.feature[elem]])
    #             # # print(form)
    #             # assert isinstance(form, Formula)
    #             #
    #             # t = (form, None, None)
    #
    #             candidates.append(t)
    #
    #     return candidates
