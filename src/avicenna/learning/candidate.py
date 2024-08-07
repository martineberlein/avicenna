from avicenna.learning.metric import (
    FitnessStrategy,
    RecallPriorityLengthFitness,
)


class Candidate:
    def __init__(self, formula, precision: float, recall: float):
        self.formula = formula
        self.precision = precision
        self.recall = recall

    def with_strategy(self, strategy: FitnessStrategy):
        return strategy.evaluate(self)

    def __eq__(self, other):
        if not isinstance(other, Candidate):
            return NotImplemented
        if self.formula == other.formula:
            return True
        else:
            strategy = RecallPriorityLengthFitness()
            return strategy.compare(self, other) == 0

    def __lt__(self, other):
        if not isinstance(other, Candidate):
            return NotImplemented
        strategy = RecallPriorityLengthFitness()
        return strategy.compare(self, other) < 0

    def __gt__(self, other):
        if not isinstance(other, Candidate):
            return NotImplemented
        strategy = RecallPriorityLengthFitness()
        return strategy.compare(self, other) > 0

    def __repr__(self):
        return f"Candidate(formula={self.formula}, precision={self.precision}, recall={self.recall})"
