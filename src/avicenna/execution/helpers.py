import re
import csv
import signal
from time import perf_counter
import bz2
import logging
from pathlib import Path

input_pattern = re.compile(r"^printf '(.*)' \|")


def read_escaped_token(line, pos):
    mode = "NORMAL"
    cp = pos + 1
    while cp < len(line):
        if line[cp] == "'" and "NORMAL" == mode:
            return line[pos : cp + 1], cp + 1
        elif mode == "ESCAPED":
            mode = "NORMAL"
        elif line[cp] == "\\" and "NORMAL" == mode:
            mode = "ESCAPED"
        cp = cp + 1
    raise AssertionError("Missing closing quotation.")


def read_unescaped_token(line, pos):
    end = line.find(" ", pos)
    if -1 == end:
        return line[pos:], len(line)
    return line[pos:end], end


def split_cli(line):
    pos = 0
    while pos < len(line):
        if "'" == line[pos]:
            token, pos = read_escaped_token(line, pos)
            yield token
        elif " " == line[pos]:
            pos = pos + 1
        else:
            token, pos = read_unescaped_token(line, pos)
            yield token


def split_grep_line(cli):
    match = re.match(input_pattern, cli)
    if match is None:
        start = 0
    else:
        start = match.end()
    stop = cli.find("timeout 1s grep", start)
    env = cli[start:stop]
    command = split_cli(cli[stop + 16 :])
    return env, list(command)


def identify_pattern(command):
    for i in range(0, len(command)):
        token = command[i]
        if token.startswith("'") and not token.startswith("'-"):
            return i
    raise AssertionError("There does not seem to be a pattern!")


class PrefixWriter:
    def __init__(self, writer, prefix):
        self.__writer = writer
        self.__prefix = prefix
        self.__remainder = ""

    def write(self, text):
        for line in text.splitlines(keepends=True):
            if line.endswith("\n"):
                self.__writer.write(self.__prefix + self.__remainder + line)
                self.__remainder = ""
            else:
                self.__remainder = self.__remainder + line

    def close(self):
        if 0 != len(self.__remainder):
            self.__writer.write(self.__prefix + self.__remainder)
            self.__remainder = ""
        self.__writer.close()

    def flush(self):
        if 0 != len(self.__remainder):
            self.__writer.write(self.__prefix + self.__remainder)
            self.__remainder = ""
        self.__writer.flush()


def time(f):
    def decorated(self, *args, **kwargs):
        start = perf_counter()
        result = f(self, *args, **kwargs)
        duration = perf_counter() - start
        self.report_performance(f.__name__, duration)
        return result

    return decorated


class Timeable:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True)
        self.__performance_file = open(self.working_dir / "performance.csv", "w")
        self.__performance_writer = csv.DictWriter(
            self.__performance_file,
            fieldnames=["iteration", "bug", "name", "time"],
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


class AlhazenTimeout(Exception):
    pass


def alhazen_timeout(_signo, _stack_frame):
    raise AlhazenTimeout()


def register_termination(timeout):
    """This throws a AlhazenTimeout within the main thread after timeout seconds."""
    if -1 != timeout:
        signal.signal(signal.SIGALRM, alhazen_timeout)
        # signal.signal(signal.SIGTERM, alhazen_timeout)
        # signal.signal(signal.SIGINT, alhazen_timeout)
        signal.alarm(timeout)


class Logging:
    def __init__(self, filename):
        fn = Path(filename)
        self.__filename = fn.parent / (str(fn.name) + ".bz2")
        self.__file = None

    def __enter__(self):
        self.__file = bz2.open(self.__filename, "at")
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s"
        )
        rootlogger = logging.getLogger("")
        rootlogger.addHandler(logging.StreamHandler())
        rootlogger.addHandler(logging.StreamHandler(self.__file))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # log the error, if any
        if exc_type is not None:
            logging.error("Uncaught exception!", exc_info=(exc_type, exc_val, exc_tb))
        # close the log file
        if self.__file is not None:
            self.__file.close()
        # re-raise exception
        return False
