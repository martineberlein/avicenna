from avicenna.learning.metric import FitnessStrategy, RecallPriorityFitness, F1ScoreFitness, PrecisionFitness, RecallFitness


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
            strategy = RecallPriorityFitness()  # or any other default strategy
            return strategy.compare(self, other) == 0

    def __lt__(self, other):
        if not isinstance(other, Candidate):
            return NotImplemented
        strategy = RecallPriorityFitness()  # or any other default strategy
        return strategy.compare(self, other) < 0

    def __gt__(self, other):
        if not isinstance(other, Candidate):
            return NotImplemented
        strategy = RecallPriorityFitness()  # or any other default strategy
        return strategy.compare(self, other) > 0

    def __repr__(self):
        return f"Candidate(precision={self.precision}, recall={self.recall})"


if __name__ == "__main__":
    # Example candidates
    candidates = [
        Candidate(formula="formula1", precision=0.75, recall=0.85),
        Candidate(formula="formula2", precision=0.70, recall=0.95),
        Candidate(formula="formula3", precision=0.80, recall=0.85),
        Candidate(formula="formula3", precision=0.60, recall=0.85),
        Candidate(formula="formula3", precision=0.62, recall=0.85),
        Candidate(formula="formula3", precision=0.60, recall=0.85),
        Candidate(formula="formula3", precision=0.95, recall=0.7),
    ]

    # Select the fitness strategy
    fitness_strategy = RecallPriorityFitness()

    # Sort candidates based on the selected fitness strategy
    sorted_candidates = sorted(candidates, key=lambda c: c.with_strategy(fitness_strategy), reverse=True)
    print(sorted_candidates)
