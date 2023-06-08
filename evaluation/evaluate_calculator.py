from avicenna.performance_evaluator import Evaluator
from avicenna.avicenna import Avicenna
from avicenna_formalizations.calculator import grammar, initial_inputs, prop
from avicenna_formalizations import get_pattern_file_path

from islearn.learner import InvariantLearner

import logging
from pathlib import Path

PATTERN_FILE = Path('../src/avicenna_formalizations/patterns.toml')


def learner_islearn(timeout, max_iterations):
    return InvariantLearner(
        grammar,
        prop,
        pattern_file=str(get_pattern_file_path())
    )


learner_avicenna = lambda timeout, max_iterations: Avicenna(
    grammar,
    prop,
    initial_inputs,
    max_iterations=max_iterations,
    pattern_file=get_pattern_file_path()
)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")

    evaluator = Evaluator('Heartbleed', timeout=60 * 60, repetitions=3, generators=[learner_islearn, learner_avicenna],
                          job_names=['ISLearn', 'Avicenna'])
    evaluator.run()
