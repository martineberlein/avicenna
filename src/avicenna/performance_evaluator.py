import time
from typing import Callable, List

from avicenna import Avicenna
from fuzzingbook.Grammars import Grammar
from islearn.learner import InvariantLearner
from isla.language import ISLaUnparser


class Evaluator:
    """
    The evaluation clas that runs and executes all experiments. Main method: run().
    """

    def __init__(
        self,
        name: str,
        generators: List[Callable[[int, int], Avicenna | InvariantLearner] | Grammar],
        job_names: List[str],
        repetitions: int = 10,
        timeout: int = 60 * 60,
    ):
        self.name = name
        self.generators = generators
        self.job_names = job_names
        self.repetitions = repetitions
        self.timeout = timeout

    def run(self):
        # generate evaluation data
        # repeat x experiments for constraint learning
        #       parallelize runs with pool
        #       measure time
        # analyze constraints with evaluation data set
        #
        results = self.learn_c(self.generators, self.job_names)

        # Analyze feature extraction

    def learn_c(self, generators, job_names):
        data = []
        for name, generator in zip(job_names, generators):
            g = generator(60, self.repetitions)
            data.append(self.learn_failure_constraints(g, name))

        for elem in data:
            print(elem)
        return data

    def learn_failure_constraints(
        self,
        generator: Avicenna | InvariantLearner,
        job_name: str,
        max_iterations=10,
        timeout_seconds=60 * 60,
        start_time=time.time(),
    ):
        failure_constraints = None
        if isinstance(generator, Avicenna):
            failure_constraints = generator.execute()
            # print("\n".join(map(lambda p: f"{p[1]}: " + p[0] + "\n", failure_constraints.items())))
        elif isinstance(generator, InvariantLearner):
            failure_constraints = generator.learn_invariants()
            # failure_constraints = next(iter(failure_constraints.items()))
            failure_constraints = list(
                map(
                    lambda p: (p[1], ISLaUnparser(p[0]).unparse()),
                    failure_constraints.items(),
                )
            )
            for i in failure_constraints:
                print(i)
            # print("\n".join(map(lambda p: f"{p[1]}:" + ISLaUnparser(p[0]).unparse(), failure_constraints.items())))

        data = {
            "name": job_name,
            "learnedConstraints": len(failure_constraints)
            if (failure_constraints is not None)
            else 0,
            "best_constraint": list(failure_constraints)[0]
            if (failure_constraints is not None)
            else None,
        }

        return data

    def _analyze_results(self):
        pass

    def _analyze_learned_input_elements(self):
        pass
