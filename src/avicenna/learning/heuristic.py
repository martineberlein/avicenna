import copy
import logging
import functools
from abc import ABC
from typing import List, Tuple, Dict, Optional, Iterable, Sequence, Set
import itertools

from sklearn.tree import DecisionTreeClassifier, export_text
from isla.language import ISLaUnparser
from avicenna.learning.treetools import (
    all_path,
    prediction_for_path,
)

import pandas
import numpy
from isla.evaluator import evaluate
from isla.language import Formula, ConjunctiveFormula
from islearn.learner import weighted_geometric_mean
from grammar_graph import gg
from isla import language
from debugging_framework.fuzzingbook.grammar import Grammar
from islearn.learner import InvariantLearner

from debugging_framework.input.oracle import OracleResult

from .table import Candidate
from ..data import Input

from avicenna.learning.learner import PatternCandidateLearner

# from avicenna.learning.table import AvicennaTruthTable, AvicennaTruthTableRow
from avicenna.learning.exhaustive import ExhaustivePatternCandidateLearner
from avicenna.learning.learner import TruthTablePatternCandidateLearner

logger = logging.getLogger("learner")


class DecisionTreeHeuristicPatternCandidateLearner(TruthTablePatternCandidateLearner):

    def get_best_candidates(self) -> Optional[List[Candidate]]:
        pass

    def get_candidates(self) -> Optional[List[Candidate]]:
        pass

    def learn_candidates(self, test_inputs: Iterable[Input], **kwargs) -> Optional[List[Candidate]]:
        pass


class HeuristicTreePatternCandidateLearner(ExhaustivePatternCandidateLearner):
    def _learn_invariants(
        self,
        positive_inputs: Set[Input],
        negative_inputs: Set[Input],
    ):
        sorted_positive_inputs = self.sort_and_filter_inputs(self.all_positive_inputs)
        candidates = self.get_candidates(sorted_positive_inputs)
        print(f"Number of candidates: ", len(candidates))

        self.evaluate_recall(candidates, positive_inputs)
        self.filter_candidates()
        self.evaluate_precision(negative_inputs)

        dataframe = self.build_dataframe()
        candidates = self.learn_decision_tree(dataframe)

        return candidates

    def build_dataframe(self):
        dataframe = pandas.DataFrame()

        number_inputs = len(self.precision_truth_table.rows[0].inputs) + len(
            self.recall_truth_table.rows[0].inputs
        )
        for idx, row in enumerate(self.precision_truth_table.rows):
            assert row.formula == self.recall_truth_table.rows[idx].formula
            dataframe[idx] = pandas.Series([numpy.nan] * number_inputs)
            dataframe[idx] = (
                row.eval_results + self.recall_truth_table.rows[idx].eval_results
            )

        dataframe["oracle"] = [False] * len(
            self.precision_truth_table.rows[0].inputs
        ) + [True] * len(self.recall_truth_table.rows[0].inputs)
        print(dataframe)
        return dataframe

    def learn_decision_tree(
        self, dataframe: pandas.DataFrame
    ) -> List[Tuple[Formula, None, None]]:

        sample_bug_count = len(dataframe[(dataframe["oracle"] == True)])
        print("number of bugs found: ", sample_bug_count)
        assert sample_bug_count > 0  # at least one bug triggering sample is required
        sample_count = len(dataframe)
        print(f"Learning with {sample_bug_count} failure inputs of {sample_count}")

        clf = DecisionTreeClassifier(
            min_samples_leaf=1,
            min_samples_split=2,  # minimal value
            max_features=5,
            max_depth=3,
            class_weight={
                True: (1.0 / sample_bug_count),
                False: (1.0 / (sample_count - sample_bug_count)),
            },
        )
        clf = clf.fit(dataframe.drop("oracle", axis=1), dataframe["oracle"])

        # clf = remove_unequal_decisions(clf)

        names = [
            ISLaUnparser(self.precision_truth_table.rows[idx].formula).unparse()
            for idx in dataframe.drop("oracle", axis=1).columns
        ]

        # print(grouped_rules(clf, feature_names=names))
        # print(names)
        tree_text = export_text(clf, feature_names=names)
        # print(tree_text)

        paths = all_path(clf)

        candidates: List[Tuple[Formula, None, None]] = []
        for path in paths:
            print(path, prediction_for_path(clf, path))
            if prediction_for_path(clf, path) == OracleResult.FAILING:
                precision_row = self.precision_truth_table.rows[
                    clf.tree_.feature[path[0]]
                ]
                recall_row = self.recall_truth_table.rows[clf.tree_.feature[path[0]]]

                for elem in path[1:-1]:
                    new_precision_row = self.precision_truth_table.rows[
                        clf.tree_.feature[elem]
                    ]
                    precision_row = precision_row.__and__(new_precision_row)

                    new_recall_row = self.recall_truth_table.rows[
                        clf.tree_.feature[elem]
                    ]
                    recall_row = recall_row.__and__(new_recall_row)

                self.precision_truth_table.append(precision_row)
                self.recall_truth_table.append(recall_row)
                t = (precision_row.formula, None, None)
                # form: Formula = self.precision_truth_table.rows[
                #     clf.tree_.feature[path[0]]
                # ].formula
                # assert isinstance(form, Formula)
                # for elem in path[1:-1]:
                #     new = self.precision_truth_table.rows[
                #         clf.tree_.feature[elem]
                #     ].formula
                #     assert isinstance(new, Formula)
                #     form = form.__and__(new)
                #     # print("Elem: ", names[clf.tree_.feature[elem]])
                # # print(form)
                # assert isinstance(form, Formula)
                #
                # t = (form, None, None)

                candidates.append(t)

        return candidates
