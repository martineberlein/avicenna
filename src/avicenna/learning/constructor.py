from typing import List, Optional, Set, Tuple
import copy

from isla.language import Formula
from islearn.learner import weighted_geometric_mean
from islearn.learner import InvariantLearner
from grammar_graph import gg

from avicenna.data import Input


class AtomicFormulaInstantiation(InvariantLearner):
    """
    The atomic candidate instantiation is a learner that generates candidates based on the
    patterns provided. It generates candidates based on the positive inputs and instantiates
    the patterns with the positive inputs.
    """

    def __init__(
        self,
        grammar,
        patterns: Optional[List[Formula]] = None,
    ):
        super().__init__(
            grammar=grammar,
            patterns=patterns,
            filter_inputs_for_learning_by_kpaths=False,
        )

    def construct_candidates(
        self,
        positive_inputs: Set[Input],
        exclude_nonterminals: Optional[Set[str]] = None,
    ) -> Set[Formula]:
        """
        Construct the candidates based on the positive inputs.
        :param positive_inputs:
        :param exclude_nonterminals:
        :return: the set of atomic candidates (based on the patterns and the positive inputs)
        """
        self.exclude_nonterminals = exclude_nonterminals or set()

        sorted_positive_inputs = self._sort_and_filter_inputs(positive_inputs)
        new_candidates: Set[Formula] = self._get_recall_candidates(
            sorted_positive_inputs
        )
        return new_candidates

    def _get_recall_candidates(
        self, sorted_positive_inputs: Set[Input]
    ) -> Set[Formula]:
        """
        Get the candidates based on the positive inputs.
        :param sorted_positive_inputs:
        :return:
        """
        candidates = self.generate_candidates(
            self.patterns, [inp.tree for inp in sorted_positive_inputs]
        )

        return candidates

    def _sort_and_filter_inputs(
        self,
        positive_inputs: Set[Input],
        max_number_positive_inputs_for_learning: int = 10,
    ) -> Set[Input]:
        """
        Sort and filter the inputs based on the number of uncovered paths and the length of the inputs.
        This method is used to filter the inputs that are used for learning.
        :param positive_inputs:
        :param max_number_positive_inputs_for_learning:
        :return:
        """
        p_dummy = copy.deepcopy(positive_inputs)
        sorted_positive_inputs = self._sort_inputs(
            p_dummy,
            self.filter_inputs_for_learning_by_kpaths,
            more_paths_weight=1.7,
            smaller_inputs_weight=1.0,
        )

        return set(sorted_positive_inputs[:max_number_positive_inputs_for_learning])

    def _sort_inputs(
        self,
        inputs: Set[Input],
        filter_inputs_for_learning_by_kpaths: bool,
        more_paths_weight: float = 1.0,
        smaller_inputs_weight: float = 0.0,
    ) -> List[Input]:
        """
        Sort the inputs based on the number of uncovered paths and the length of the inputs.
        """
        assert more_paths_weight or smaller_inputs_weight
        result: List[Input] = []

        tree_paths = {
            inp: {
                path
                for path in self.graph.k_paths_in_tree(inp.tree.to_parse_tree(), self.k)
                if (
                    not isinstance(path[-1], gg.TerminalNode)
                    or (
                        not isinstance(path[-1], gg.TerminalNode)
                        and len(path[-1].symbol) > 1
                    )
                )
            }
            for inp in inputs
        }

        covered_paths: Set[Tuple[gg.Node, ...]] = set([])
        max_len_input = max(len(inp.tree) for inp in inputs)

        def uncovered_paths(inp: Input) -> Set[Tuple[gg.Node, ...]]:
            return {path for path in tree_paths[inp] if path not in covered_paths}

        def sort_by_paths_key(inp: Input) -> float:
            return len(uncovered_paths(inp))

        def sort_by_length_key(inp: Input) -> float:
            return len(inp.tree)

        def sort_by_paths_and_length_key(inp: Input) -> float:
            return weighted_geometric_mean(
                [len(uncovered_paths(inp)), max_len_input - len(inp.tree)],
                [more_paths_weight, smaller_inputs_weight],
            )

        if not more_paths_weight:
            key = sort_by_length_key
        elif not smaller_inputs_weight:
            key = sort_by_paths_key
        else:
            key = sort_by_paths_and_length_key

        while inputs:
            inp = sorted(inputs, key=key, reverse=True)[0]
            inputs.remove(inp)
            uncovered = uncovered_paths(inp)

            if filter_inputs_for_learning_by_kpaths and not uncovered:
                continue

            covered_paths.update(uncovered)
            result.append(inp)

        return result
