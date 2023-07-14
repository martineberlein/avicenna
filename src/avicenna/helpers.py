from time import perf_counter
import csv
import signal
import pandas
from typing import Set

from islearn.learner import InvariantLearner
from isla.evaluator import evaluate

from avicenna.oracle import OracleResult
from avicenna.islearn import AvicennaISlearn
from avicenna.input import Input


def constraint_eval(test_inputs: Set[Input], constraint, grammar):
    data = []
    for inp in test_inputs:
        data.append(
            {
                "name": str(inp),
                "predicted": evaluate(constraint, inp.tree, grammar),
                "oracle": True if inp.oracle == OracleResult.BUG else False,
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


def instantiate_learner(
    grammar,
    oracle,
    activated_patterns=None,
    pattern_file=None,
    min_recall=0.9,
    min_specificity=0.6,
    deactivated_patterns: Set = None,
    max_conjunction_size=2,
) -> AvicennaISlearn:
    """
    Helper function that calls ISLearn to learn a set of invariants from a given set of input samples.
    :param max_conjunction_size:
    :param grammar:
    :param oracle:
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
    return AvicennaISlearn(
        grammar,
        oracle,
        activated_patterns=activated_patterns,
        reduce_inputs_for_learning=False,
        reduce_all_inputs=False,
        filter_inputs_for_learning_by_kpaths=False,
        do_generate_more_inputs=False,
        generate_new_learning_samples=False,
        min_specificity=min_specificity,
        min_recall=min_recall,
        pattern_file=pattern_file,
        deactivated_patterns=deactivated_patterns,
        max_conjunction_size=max_conjunction_size,
        # target_number_positive_samples=20,
        target_number_positive_samples_for_learning=10
    )


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
