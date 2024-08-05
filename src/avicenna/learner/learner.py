from typing import Iterable, List, Optional, Set

from debugging_framework.fuzzingbook.grammar import Grammar
from isla.language import Formula

from avicenna.input.input import Input


class PatternLearner:
    def __init__(
        self,
        grammar: Grammar,
        pattern_file: Optional[str] = None,
        patterns: Optional[List[Formula]] = None,
    ):
        pass

    def learn_failure_invariants(
        self,
        test_inputs: Set[Input],
    ):
        raise NotImplementedError()
