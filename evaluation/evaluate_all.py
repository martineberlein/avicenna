from typing import List, Tuple, Callable, Set

from fuzzingbook.Grammars import Grammar

from avicenna_formalizations.calculator import (
    oracle as calculator_oracle,
    initial_inputs as calculator_initial_inputs,
    grammar as calculator_grammar
)
from avicenna_formalizations.heartbeat import (
    oracle as heartbeat_oracle,
    initial_inputs as heartbeat_initial_inputs,
    grammar as heartbeat_grammar
)

if __name__ == "__main__":

    subjects: List[Tuple[str ,Callable, List, Grammar]] = [
        ("Calculator", calculator_oracle, calculator_initial_inputs, calculator_grammar),
        ("Heartbleed", heartbeat_oracle, heartbeat_initial_inputs, heartbeat_grammar),
    ]

    for subject in subjects:
        name, oracle, initial_inputs, grammar = subject
