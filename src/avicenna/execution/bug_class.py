from abc import ABC, abstractmethod
from src.avicenna.execution import oracles
from pathlib import Path
from typing import Generator
from pandas import DataFrame
from typing import Union
import importlib
import sys
from typing import Any


class Bug(ABC):
    @abstractmethod
    def subject(self) -> str:
        """:return a name for the program under test"""
        raise AssertionError("Overwrite in subclass.")

    @abstractmethod
    def execute_sample_list(self, execdir, samples) -> DataFrame:
        """:return a DataFrame which contains one row for each sample."""
        raise AssertionError("Overwrite in subclass.")

    @abstractmethod
    def grammar_file(self) -> Path:
        """:return the path to the grammar to be used."""
        raise AssertionError("Overwrite in subclass.")

    @abstractmethod
    def sample_files(self) -> Generator[Path, None, None]:
        """A generator methods which yields the sample files to work with for this bug."""
        raise AssertionError("Overwrite in subclass.")

    def tear_down(self):
        """You can overwrite this if there is some cleanup to be done after an alhazen run for this bug."""
        pass

    def suffix(self) -> str:
        """:return the suffix for input files for this program under test."""
        g = self.sample_files()
        sample = next(g)
        return sample.suffix

    def execute_samples(self, sample_dir) -> DataFrame:
        """helper method to execute all samples in a given directory."""
        return self.execute_sample_list(sample_dir.parent, list(sample_dir.iterdir()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tear_down()


def load_driver(file: Union[Path, str]) -> Bug:
    file = Path(file)
    # add the parent dir of the module to the system path
    # this makes sure that the module can load file which are next to it in the file system
    sys.path.append(str(file.parent.resolve()))
    loader = importlib.machinery.SourceFileLoader(
        file.name[: file.name.rfind(".")], str(file)
    )
    # typing this as Any means that pyre will not complain about the create_bug() method
    drivermodule: Any = loader.load_module()
    bug = drivermodule.create_bug()
    assert bug is not None, "The driver did not provide a bug."
    return bug
