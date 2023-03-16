from time import perf_counter
import csv
import signal
import pandas
from typing import Dict, Set

from islearn.learner import InvariantLearner
from isla.evaluator import evaluate
from isla.language import ISLaUnparser


def constraint_eval(trees, oracle, constraint, grammar):
    data = []
    for tree, o in zip(trees, oracle):
        data.append(
            {
                "name": str(tree),
                "predicted": evaluate(constraint, tree, grammar),
                "oracle": o,
            }
        )

    return pandas.DataFrame.from_records(data)


class CustomTimeout(Exception):
    pass


def custom_timeout(_signo, _stack_frame):
    raise CustomTimeout()


def register_termination(timeout):
    """This throws a AlhazenTimeout within the main thread after timeout seconds."""
    if -1 != timeout:
        signal.signal(signal.SIGALRM, custom_timeout)
        # signal.signal(signal.SIGTERM, alhazen_timeout)
        # signal.signal(signal.SIGINT, alhazen_timeout)
        signal.alarm(timeout)


def run_islearn(
    grammar,
    prop,
    positive_trees,
    negative_trees=None,
    activated_patterns=None,
    excluded_features=None,
    pattern_file=None,
    min_recall=0.9,
    min_specificity=0.6,
    deactivated_patterns: Set = None,
    max_conjunction_size=2,
):
    """
    Helper function that calls ISLearn to learn a set of invariants from a given set of input samples.
    :param grammar:
    :param prop:
    :param positive_trees:
    :param negative_trees:
    :param activated_patterns:
    :param excluded_features:
    :param pattern_file:
    :param min_recall:
    :param min_specificity:
    :param deactivated_patterns: Set = None
:
    :return:

    """
    # Start ISLearn with the InvariantLearner
    result = InvariantLearner(
        grammar,
        prop,
        activated_patterns=activated_patterns,
        positive_examples=positive_trees,
        negative_examples=negative_trees,
        exclude_nonterminals=excluded_features,
        reduce_inputs_for_learning=False,
        # reduce_all_inputs=False,
        # filter_inputs_for_learning_by_kpaths=False,
        do_generate_more_inputs=False,
        generate_new_learning_samples=False,
        min_specificity=min_specificity,
        min_recall=min_recall,
        pattern_file=pattern_file,
        deactivated_patterns=deactivated_patterns,
        max_conjunction_size=1,
        # target_number_positive_samples=20,
        # target_number_positive_samples_for_learning=20
    ).learn_invariants()

    constraints = list(
        map(lambda p: f"" + ISLaUnparser(p[0]).unparse(), result.items())
    )

    return constraints


def time(f):
    def decorated(self, *args, **kwargs):
        start = perf_counter()
        result = f(self, *args, **kwargs)
        duration = perf_counter() - start
        self.report_performance(f.__name__, duration)
        return result

    return decorated


class Timetable:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True)
        self.__performance_file = open(self.working_dir / "performance.csv", "w")
        self.__performance_writer = csv.DictWriter(
            self.__performance_file,
            fieldnames=["iteration", "name", "time"],
            dialect="unix",
        )
        self.__performance_writer.writeheader()

    def report_performance(self, name, duration):
        self.__performance_writer.writerow(
            dict({"name": name, "time": duration}, **self.iteration_identifier_map())
        )

    def iteration_identifier_map(self):
        raise AssertionError("Overwrite in subclass.")

    def finalize_performance(self):
        # write performance data to disk
        self.__performance_file.close()
