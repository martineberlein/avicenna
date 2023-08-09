from time import perf_counter
import csv
import signal
import pandas
from typing import Set

from islearn.learner import InvariantLearner
from isla.evaluator import evaluate

from avicenna.oracle import OracleResult
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
