import copy
from typing import List, Dict, Iterable, Sequence, Set, Optional

from isla.evaluator import evaluate
from grammar_graph import gg
from isla import language
from debugging_framework.input.oracle import OracleResult
from avicenna.input.input import Input
from avicenna.learning.metric import FitnessStrategy, RecallPriorityLengthFitness


class Candidate:
    """
    A candidate contains a formula, a set of inputs, a list of evaluation results and a combination of inputs and
    evaluation results.
    """

    def __init__(
        self,
        formula: language.Formula,
        inputs: Set[Input] = None,
        positive_eval_results: Sequence[bool] = (),
        negative_eval_results: Sequence[bool] = (),
        comb: Dict[Input, bool] = None,
    ):
        """
        Initialize a candidate with a formula, a set of inputs, a list of evaluation results and a combination of inputs
        and evaluation results.
        :param Formula formula: The formula of the candidate.
        :param Set[Input] inputs: The inputs of the candidate.
        :param Sequence[bool] positive_eval_results: The evaluation results of the candidate on positive inputs.
        :param Sequence[bool] negative_eval_results: The evaluation results of the candidate on negative inputs.
        :param Dict[Input, bool] comb: The combination of inputs and evaluation results.
        """
        self.formula = formula
        self.inputs = inputs or set()
        self.failing_inputs_eval_results: List[bool] = list(positive_eval_results)
        self.passing_inputs_eval_results: List[bool] = list(negative_eval_results)
        self.comb: Dict[Input, bool] = comb or {}

    def __copy__(self):
        return Candidate(
            self.formula, self.inputs, self.failing_inputs_eval_results,
            self.passing_inputs_eval_results, self.comb
        )

    def evaluate(
        self,
        test_inputs: Set[Input],
        graph: gg.GrammarGraph,
    ):
        """
        Evaluate the formula on a set of inputs and update the evaluation results and combination.
        """
        new_inputs = test_inputs - self.inputs

        for inp in new_inputs:
            eval_result = evaluate(
                self.formula, inp.tree, graph.grammar, graph=graph
            ).is_true()
            self._update_eval_results_and_combination(eval_result, inp)

        self.inputs.update(new_inputs)
        assert self.inputs_are_valid()

    def _update_eval_results_and_combination(self, eval_result: bool, inp: Input):
        """
        Update the evaluation results and combination with a new input and its evaluation result.
        """
        if inp.oracle == OracleResult.FAILING:
            self.failing_inputs_eval_results.append(eval_result)
        else:
            self.passing_inputs_eval_results.append(eval_result)
        self.comb[inp] = eval_result

    def specificity(self) -> float:
        """
        Return the specificity of the candidate.
        """
        return sum(not int(entry) for entry in self.passing_inputs_eval_results) / len(self.passing_inputs_eval_results)

    def recall(self) -> float:
        """
        Return the recall of the candidate.
        """
        return sum(int(entry) for entry in self.failing_inputs_eval_results) / len(self.failing_inputs_eval_results)

    def precision(self) -> float:
        """
        Return the precision of the candidate.
        """
        tp = sum(int(entry) for entry in self.failing_inputs_eval_results)
        fp = sum(int(entry) for entry in self.passing_inputs_eval_results)
        return tp / (tp + fp) if tp + fp > 0 else 0.0

    def inputs_are_valid(self) -> bool:
        """
        Return whether the candidate has valid inputs and evaluation results.
        """
        return len(self.inputs) == (len(self.failing_inputs_eval_results) + len(self.passing_inputs_eval_results)) and all(
            isinstance(entry, bool) for entry in self.failing_inputs_eval_results + self.passing_inputs_eval_results
        )

    def with_strategy(self, strategy: FitnessStrategy):
        """
        Return the evaluation of the candidate with a given fitness strategy.
        """
        return strategy.evaluate(self)

    def __lt__(self, other):
        """
        Return whether a candidate is less than another candidate based on a fitness strategy.
        """
        if not isinstance(other, Candidate):
            return NotImplemented
        strategy = RecallPriorityLengthFitness()
        return strategy.compare(self, other) < 0

    def __gt__(self, other):
        """
        Return whether a candidate is greater than another candidate based on a fitness strategy.
        """
        if not isinstance(other, Candidate):
            return NotImplemented
        strategy = RecallPriorityLengthFitness()
        return strategy.compare(self, other) > 0

    def __repr__(self):
        """
        Return a string representation of the candidate.
        """
        return f"Candidate({str(self.formula)},failing={repr(self.failing_inputs_eval_results)}, passing={repr(self.passing_inputs_eval_results)})"

    def __str__(self):
        """
        Return a string representation of the candidate.
        """
        return self.__repr__()

    def __eq__(self, other):
        """
        Return whether two candidates are equal.
        """
        return (
            isinstance(other, Candidate) and self.formula == other.formula
        )

    def __len__(self):
        """
        Return the number of inputs in the candidate.
        """
        return len(self.inputs)

    def __hash__(self):
        """
        Return a hash of the candidate based on its formula.
        """
        return hash(self.formula)

    def __neg__(self):
        """
        Return the negation of the candidate formula.
        """
        comb = {}
        for inp in self.comb.keys():
            comb[inp] = not self.comb[inp]

        return Candidate(
            formula=-self.formula,
            inputs=self.inputs,
            positive_eval_results=[not eval_result for eval_result in self.failing_inputs_eval_results],
            negative_eval_results=[not eval_result for eval_result in self.passing_inputs_eval_results],
            comb=comb,
        )

    def __and__(self, other: "Candidate") -> "Candidate":
        """
        Return the conjunction of two candidates by combining their formulas, inputs and evaluation results.
        """
        assert len(self.inputs) == len(other.inputs)
        assert len(self.failing_inputs_eval_results) == len(other.failing_inputs_eval_results)
        assert len(self.passing_inputs_eval_results) == len(other.passing_inputs_eval_results)

        new_failing_results = []
        new_passing_results = []
        comb = {}
        for inp in self.comb.keys():
            r = self.comb[inp] and other.comb[inp]
            if inp.oracle == OracleResult.FAILING:
                new_failing_results.append(r)
            else:
                new_passing_results.append(r)
            comb[inp] = r

        inputs = copy.copy(self.inputs)

        return Candidate(
            formula=self.formula & other.formula,
            inputs=inputs,
            positive_eval_results=new_failing_results,
            negative_eval_results=new_passing_results,
            comb=comb,
        )

    def __or__(self, other: "Candidate") -> "Candidate":
        """
        Return the disjunction of two candidates by combining their formulas, inputs and evaluation results.
        """
        raise NotImplementedError()


class CandidateSet:
    """
    A truth table is a list of truth table rows. It is used to store the results of evaluating formulas on a set of inputs.
    """

    def __init__(self, candidates: Iterable[Candidate] = ()):
        self.candidate_hashes = set()
        self.candidates = []
        for candidate in candidates:
            candidate_hash = hash(candidate)
            if candidate_hash not in self.candidate_hashes:
                self.candidate_hashes.add(candidate_hash)
                self.candidates.append(candidate)

    def __deepcopy__(self, memodict=None):
        """
        Return a deep copy of the candidate set and its candidates.
        """
        return CandidateSet([copy.copy(row) for row in self.candidates])

    def __repr__(self):
        """
        Return a string representation of the candidate set and its candidates.
        """
        return f"CandidateSet({repr(self.candidates)})"

    def __str__(self):
        """
        Return a string representation of the candidate set and its candidates.
        """
        return "\n".join(map(str, self.candidates))

    def __getitem__(self, item: int | language.Formula) -> Optional[Candidate]:
        """
        Retrieve a candidate by index or formula.
        """
        if isinstance(item, int):
            return self.candidates[item]
        assert isinstance(item, language.Formula)
        try:
            return next(row for row in self.candidates if row.formula == item)
        except StopIteration:
            return None

    def __len__(self):
        """
        Return the number of candidates in the candidate set.
        """
        return len(self.candidates)

    def __iter__(self):
        """
        Iterate over the candidates in the candidate set.
        """
        return iter(self.candidates)

    def append(self, candidate: Candidate):
        """
        Add a candidate to the candidate set.
        """
        candidate_hash = hash(candidate)
        if candidate_hash not in self.candidate_hashes:
            self.candidate_hashes.add(candidate_hash)
            self.candidates.append(candidate)

    def remove(self, candidate: Candidate):
        """
        Remove a candidate from the candidate set.
        """
        candidate_hash = hash(candidate)
        if candidate_hash in self.candidate_hashes:
            self.candidates.remove(candidate)
            self.candidate_hashes.remove(candidate_hash)

    def __add__(self, other: "CandidateSet") -> "CandidateSet":
        """
        Combine two candidate sets.
        """
        return CandidateSet(self.candidates + other.candidates)