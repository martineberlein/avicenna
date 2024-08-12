from typing import Set, Iterable, Optional, List
from abc import ABC, abstractmethod

from islearn.mutation import MutationFuzzer
from isla.fuzzer import GrammarFuzzer
from isla.language import DerivationTree
from isla.solver import ISLaSolver
from debugging_framework.fuzzingbook.grammar import Grammar
from debugging_framework.fuzzingbook.fuzzer import (
    GrammarFuzzer as FuzzingbookGrammarFuzzer,
)

from ..data import Input, OracleResult
from ..learning.table import Candidate


class Generator(ABC):
    """
    A generator is responsible for generating inputs to be used in the debugging process.
    """

    def __init__(self, grammar: Grammar, **kwargs):
        """
        Initialize the generator with a grammar.
        """
        self.grammar = grammar

    @abstractmethod
    def generate(self, *args, **kwargs) -> Input:
        """
        Generate an input to be used in the debugging process.
        """
        raise NotImplementedError

    def generate_test_inputs(self, num_inputs: int = 10, **kwargs) -> Set[Input]:
        """
        Generate multiple inputs to be used in the debugging process.
        """
        test_inputs = set()
        for _ in range(num_inputs):
            inp = self.generate(**kwargs)
            if inp:
                test_inputs.add(inp)
        return test_inputs

    def reset(self, **kwargs):
        """
        Reset the generator.
        """
        pass


class FuzzingbookBasedGenerator(Generator):
    """
    A generator that uses the fuzzingbook grammar fuzzer to generate inputs.
    """

    def __init__(self, grammar: Grammar, **kwargs):
        super().__init__(grammar)
        self.fuzzer = FuzzingbookGrammarFuzzer(grammar)

    def generate(self, **kwargs) -> Input:
        """
        Generate an input to be used in the debugging process.
        """
        return Input(self.fuzzer.fuzz_tree())


class ISLaGrammarBasedGenerator(Generator):
    """
    A generator that uses the ISLa Grammar-based Fuzzer to generate inputs.
    This generator directly produces the derivation trees, which is more efficient than the FuzzingbookBasedGenerator.
    """

    def __init__(self, grammar: Grammar, **kwargs):
        super().__init__(grammar)
        self.fuzzer = GrammarFuzzer(grammar, max_nonterminals=20)

    def generate(self, **kwargs) -> Input:
        """
        Generate an input to be used in the debugging process.
        """
        return Input(tree=self.fuzzer.fuzz_tree())


class ISLaSolverGenerator(Generator):
    """
    A generator that uses the ISLa Solver to generate inputs.
    """

    def __init__(self, grammar: Grammar, enable_optimized_z3_queries=False, **kwargs):
        super().__init__(grammar)
        self.solver: Optional[ISLaSolver] = ISLaSolver(self.grammar)
        self.enable_optimized_z3_queries = enable_optimized_z3_queries

    def generate(self, **kwargs) -> Optional[Input]:
        """
        Generate an input to be used in the debugging process using the ISLa Solver.
        """
        try:
            tree = self.solver.solve()
            return Input(tree=tree)
        except (StopIteration, RuntimeError):
            return None

    def generate_test_inputs(
        self, num_inputs: int = 10, candidates: List[Candidate] = None, **kwargs
    ) -> Set[Input]:
        """
        Generate multiple inputs to be used in the debugging process.
        """
        test_inputs = set()

        for candidate in candidates:
            self.reset(candidate.formula)
            for _ in range(num_inputs):
                inp = self.generate(**kwargs)
                if inp:
                    test_inputs.add(inp)
        return test_inputs

    def reset(self, constraint, enable_optimized_z3_queries=False, **kwargs):
        """
        Reset the generator with a new constraint.
        """
        self.solver = ISLaSolver(
            self.grammar,
            constraint,
            enable_optimized_z3_queries=enable_optimized_z3_queries,
        )


class MutationBasedGenerator(Generator):
    """
    A generator that uses the Mutation Fuzzer to generate inputs based on a seed.
    """

    def __init__(
        self,
        grammar: Grammar,
        oracle,
        initial_inputs: Set[Input],
        yield_negative: bool = False,
        **kwargs,
    ):
        super().__init__(grammar)
        self.yield_negative = yield_negative
        self.oracle = oracle
        self.seed = [inp.tree for inp in initial_inputs]
        self.fuzzer = self.AvicennaMutationFuzzer(grammar, self.seed, oracle).run(
            yield_negative=self.yield_negative
        )

    class AvicennaMutationFuzzer(MutationFuzzer):
        def __init__(self, grammar: Grammar, seed: Iterable[DerivationTree], oracle):
            super().__init__(grammar, seed)
            self.property = oracle

        def process_new_input(
            self, inp: DerivationTree, extend_fragments: bool = True
        ) -> bool:
            new_coverage = self.coverages_seen - self.coverages_of(inp)
            result = self.property(inp)
            oracle_result: OracleResult = result if isinstance(result, OracleResult) else result[0]
            if (
                inp in self.population
                or not oracle_result.is_failing()
                or not new_coverage
            ):
                return False

            self.coverages_seen.update(new_coverage)
            self.population.add(inp)
            if extend_fragments:
                self.update_fragments(inp)

            return True

    def generate(self, **kwargs) -> Optional[Input]:
        """
        Generate an input to be used in the debugging process using the Mutation Fuzzer.
        """
        try:
            return Input(tree=next(self.fuzzer))
        except StopIteration:
            return None

    def reset(self, seed, **kwargs):
        """
        Reset the generator with a new seed.
        """
        self.seed = [inp.tree for inp in seed]
        self.fuzzer = self.AvicennaMutationFuzzer(
            self.grammar, self.seed, self.oracle
        ).run(yield_negative=self.yield_negative)
