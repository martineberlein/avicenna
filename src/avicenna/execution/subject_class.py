import abc
from abc import abstractmethod
from pathlib import Path
from typing import Union
import pandas
import json
import importlib.util
import sys
import numpy
from typing import Dict, List
import tempfile

from itertools import zip_longest

from src.avicenna.execution.external_exec import call_java
from src.avicenna.execution.oracle import OracleResult
from src.avicenna.execution.helper import get_all_files


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return [[c for c in chunk if c is not None] for chunk in zip_longest(*args)]


class Subject(metaclass=abc.ABCMeta):
    def __init__(self, error):
        self.__error = error

    @abstractmethod
    def subject(self) -> str:
        """:return a name for the program under test"""
        raise AssertionError("Overwrite in subclass.")

    @abstractmethod
    def jar_location(self) -> str:
        """:return the absolute path to the subject jar"""
        raise AssertionError("Overwrite in subclass.")

    @abstractmethod
    def grammar(self) -> Dict:
        """:return the grammar"""
        raise AssertionError("Overwrite in subclass.")

    def execute_sample_strings(self, samples: List[str], tmp_dir: Path = None):
        if tmp_dir is None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                return self.__execute_sample_strings(
                    tmp_dir=Path(tmp_dir), samples=samples
                )
        else:
            return self.__execute_sample_strings(tmp_dir=tmp_dir, samples=samples)

    def __execute_sample_strings(self, samples: List[str], tmp_dir: Path = None):
        sample_paths: List[Path] = []
        for i, item in enumerate(samples):
            sample_dir = Path(tmp_dir) / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            path = sample_dir / "file_{}.txt".format(i)
            with open(path, "w") as f:
                f.write(str(item))
                f.close()
            sample_paths.append(path)

        return self.execute_sample_list(
            execdir=tmp_dir, samples=sample_paths, sample_content=samples
        )

    def execute_sample_list(self, execdir, samples, sample_content):
        jarfile = self.jar_location()
        exception_log = execdir / "exceptions.log"
        execution_log = execdir / "execution.log"

        # per_chunk = []
        # for chunk in grouper(samples, 1000):
        cmd = [
            "-jar",
            jarfile,
            "--ignore-exceptions",
            "--log-exceptions",
            exception_log,
        ] + samples
        call_java(cmd, execution_log, cwd=execdir)

        exceptions_data = self.read_exception_log(exception_log)
        all_data = self.sample_frame(samples, sample_content=sample_content)

        joined = (
            exceptions_data.set_index("file")
            .join(all_data.set_index("file"), how="outer")
            .reset_index()
        )
        # per_chunk.append(self.__apply_oracle(joined))

        return self.__apply_oracle(joined)

    def execute_samples_dir(self, sample_dir):
        """:return a DataFrame with the execution data"""
        exception_log = Path(sample_dir).parent / "exceptions.log"
        execution_log = Path(sample_dir).parent / "execution.log"

        cmd = [
            "-jar",
            self.jar_location(),
            "--ignore-exceptions",
            "--log-exceptions",
            str(exception_log.absolute()),
            sample_dir,
        ]
        call_java(cmd, execution_log)

        exceptions_data = self.read_exception_log(exception_log)
        sample_frame = self.sample_frame_dir(sample_dir)

        joined_data = (
            exceptions_data.set_index("file")
            .join(sample_frame.set_index("file"), how="outer")
            .reset_index()
        )

        return self.__apply_oracle(joined_data)

    def read_exception_log(self, exception_log):
        if not exception_log.exists():
            return pandas.DataFrame(columns=["file", "hash", "exception", "location"])
        else:
            with open(exception_log, "r") as except_in:
                exceptions = json.load(except_in)
            exceptions_data = []
            for exp in exceptions:
                stack_hash = exp["stack_hash"]
                exception_name = exp["name"]
                exception_location = exp["location"]
                for file in exp["files"]:
                    exceptions_data.append(
                        {
                            "file": Path(file).name,
                            "hash": stack_hash,
                            "exception": exception_name,
                            "location": exception_location,
                        }
                    )
            exceptions_data = pandas.DataFrame.from_records(exceptions_data)
            return exceptions_data

    def sample_frame_dir(self, sample_dir):
        samples = get_all_files(sample_dir)
        return self.sample_frame(samples)  # TODO Read from files and safe in Dataframe

    def sample_frame(self, samples, sample_content):
        all_data = []
        for i, file in enumerate(samples):
            all_data.append(
                {
                    "file": str(Path(file).name),
                    "sample": str(sample_content[i]),
                    "subject": str(self.subject()),
                }
            )
        if 0 == len(all_data):
            return pandas.DataFrame(columns=["file", "subject"])
        return pandas.DataFrame.from_records(all_data)

    def __apply_oracle(self, exceptions_data):
        exceptions_data["oracle"] = numpy.where(
            exceptions_data["hash"] == self.__error,
            OracleResult.BUG,
            OracleResult.NO_BUG,
        )
        return exceptions_data


def load_driver(file: Union[Path, str]) -> Subject:
    file = Path(file)
    spec = importlib.util.spec_from_file_location("create_bug", str(file.resolve()))
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(file.resolve())] = module
    spec.loader.exec_module(module)
    return module.create_bug()
