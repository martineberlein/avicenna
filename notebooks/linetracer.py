from typing import Optional, Dict, Set, Tuple, List

from abc import ABC, abstractmethod
import os
import subprocess
from pathlib import Path
import coverage


COVERAGE = "coverage"


class Report:
    """A report of the coverage data."""

    def __init__(self, command: str):
        self.command: str = command
        self.successful: Optional[bool] = None
        self.raised_exception: Optional[Exception] = None

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CoverageReport(Report):
    """A report of the coverage data."""

    def __init__(
        self,
        coverage_data: Dict[str, Set[int]],
        total_executable_lines: Dict[str, Set[int]],
    ):
        super().__init__(COVERAGE)
        self.coverage_data = coverage_data
        self.total_executable_lines = total_executable_lines

    def get_file_coverage(self, file: str) -> float:
        """Get the coverage of a file."""

        if file not in self.coverage_data:
            return 0.0

        total_lines = len(self.total_executable_lines[file])
        covered_lines = len(self.coverage_data[file])
        return covered_lines / total_lines

    def get_total_coverage(self) -> float:
        """Get the total coverage of the project."""

        total_lines = sum(len(lines) for lines in self.total_executable_lines.values())
        covered_lines = sum(len(lines) for lines in self.coverage_data.values())
        return covered_lines / total_lines

    def __repr__(self):
        return (
            f"CoverageReport("
            f"{[str(Path(file).name) + ": " + str(lines) for file, lines in self.coverage_data.items()]}, "
            f"coverage={self.get_total_coverage()})"
        )


class CoverageAnalyzer(ABC):
    """Analyze coverage data."""

    def __init__(
        self,
        project_root: os.PathLike,
        test: List[str],
    ):
        self.project_root = project_root
        self.test: List[str] = test

    @abstractmethod
    def get_coverage(self):
        """Run the tests with coverage and analyze the results."""
        pass


class CoveragePyAnalyzer(CoverageAnalyzer):
    """Analyze coverage data using the coverage.py module."""

    def __init__(
        self,
        project_root: os.PathLike,
        test: str | List[str],
        harness: Optional[os.PathLike] = None,
        output: Optional[os.PathLike] = None,
    ):
        tests = test if isinstance(test, list) else [test]
        super().__init__(project_root, tests)
        self.output = output if output else Path(project_root) / ".coverage"
        self.harness = harness if harness else Path(project_root) / "harness.py"

    @staticmethod
    def get_all_executable_lines(cov: coverage.Coverage) -> Dict[str, Set[int]]:
        """Get all executable lines from the coverage data."""
        all_executable_lines = dict()
        data = cov.get_data()
        for file in data.measured_files():
            analysis = cov.analysis2(file)
            executable_lines = analysis[1]
            all_executable_lines[file] = set(executable_lines)
        return all_executable_lines

    def analyze_coverage_data(self) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]]]:
        """Analyze the coverage data."""
        coverage_data = dict()
        coverage_file = self.output

        cov = coverage.Coverage(data_file=coverage_file)
        cov.load()
        data = cov.get_data()
        print(cov.report())

        for file in data.measured_files():
            if coverage_data.get(file) is None:
                coverage_data[file] = set(data.lines(file))
            else:
                coverage_data[file].update(data.lines(file))

        total_executable_lines = self.get_all_executable_lines(cov)
        return coverage_data, total_executable_lines

    def run_coverage_for_test(self, command: List[str], test: str):
        result = subprocess.run(
            command + [self.harness, test],
            text=True,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        print(result.stdout)
        return result

    def clean_coverage(self):
        subprocess.run(["coverage", "erase", f"--data-file={self.output}",], cwd=self.project_root)

    def get_coverage(self, clean: bool = True) -> CoverageReport:
        """Run the tests with coverage and analyze the results."""

        if clean:
            self.clean_coverage()

        command = [
            "coverage",
            "run",
            f"--data-file={self.output}",
            f"--source={self.project_root}",
        ]
        if len(self.test) > 1:
            command.append("--append")

        for test in self.test:
            result = self.run_coverage_for_test(command, test)

        coverage_data, total_executable_lines = self.analyze_coverage_data()

        return CoverageReport(
            coverage_data,
            total_executable_lines
        )
