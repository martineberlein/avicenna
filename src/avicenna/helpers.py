from time import perf_counter
import csv
import signal
import pandas
from typing import Set

from islearn.learner import InvariantLearner
from isla.evaluator import evaluate

from avicenna.oracle import OracleResult
from avicenna.input import Input


def map_to_bool(result: OracleResult) -> bool:
    match result:
        case OracleResult.BUG:
            return True
        case OracleResult.NO_BUG:
            return False
        case _:
            return False
