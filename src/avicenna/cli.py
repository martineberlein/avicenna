from typing import Tuple, List, Dict, Iterable, Any, cast
import argparse
from argparse import Namespace, ArgumentParser
import sys
import logging
from io import TextIOWrapper
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from avicenna import Avicenna, __version__ as avicenna_version

Grammar = Dict[str, List[str]]

GRAMMAR_ERROR = -2


def main(*args: str, stdout=sys.stdout, stderr=sys.stderr):
    parser = create_parser()

    args = parser.parse_args(args or sys.argv[1:])

    if len(sys.argv[1:]) < 1 and not args.version:
        parser.print_usage(file=stderr)
        print(
            "Avicenna: error: To use avicenna you need to provide a grammar, a set of initial input samples, "
            "and an oracle."
        )
        exit(0)

    if args.version:
        print(f"Avicenna version {avicenna_version}", file=stdout)
        sys.exit(0)

    level_mapping = {
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    if hasattr(args, "log_level"):
        logging.basicConfig(stream=stderr, level=level_mapping[args.log_level])

    grammar = parse_grammar(Path(args.grammar).resolve(), stderr)
    execute_sample = parse_prop(Path(args.grammar).resolve())

    try:
        avicenna = Avicenna(
            grammar=grammar,
            evaluation_function=args.prop,
            initial_inputs=args.initial_inputs,
        )
        avicenna.execute()
    except KeyboardInterrupt:
        sys.exit(0)


def parse_grammar(grammar_file, stderr) -> Grammar:
    try:
        grammar = {}
        with open(grammar_file, "r") as grammar_file:
            with redirect_stderr(stderr):
                grammar_file_content = grammar_file.read()
                grammar |= process_python_extension(grammar_file_content, stderr)

        if not grammar:
            print(
                "Could not find any grammar definition in the given files.",
                file=stderr,
            )
            sys.exit(GRAMMAR_ERROR)

    except Exception as exc:
        exc_string = str(exc)
        if exc_string == "None":
            exc_string = ""
        print(
            f"avicenna: error: A {type(exc).__name__} occurred "
            + "while processing a provided file"
            + (f" ({exc_string})" if exc_string else ""),
            file=stderr,
        )
        sys.exit(GRAMMAR_ERROR)

    return grammar


def parse_prop(prop_file, stderr) -> Grammar:
    try:
        prop = {}
        with open(prop_file, "r") as prop_file:
            with redirect_stderr(stderr):
                grammar_file_content = prop_file.read()
                prop |= process_python_extension(grammar_file_content, stderr)

        if not prop:
            print(
                "Could not find any grammar definition in the given files.",
                file=stderr,
            )
            sys.exit(GRAMMAR_ERROR)

    except Exception as exc:
        exc_string = str(exc)
        if exc_string == "None":
            exc_string = ""
        print(
            f"avicenna: error: A {type(exc).__name__} occurred "
            + "while processing a provided file"
            + (f" ({exc_string})" if exc_string else ""),
            file=stderr,
        )
        sys.exit(GRAMMAR_ERROR)

    return grammar


def process_python_extension(python_file_content: str, stderr) -> Grammar:
    query_program = """
try:
    grammar_ = grammar() if callable(grammar) else grammar
except NameError:
    grammar_ = None

try:
    predicates_ = predicates() if callable(predicates) else None
except NameError as err:
    predicates_ = None
    err_ = err
"""

    python_file_content = f"{python_file_content}\n{query_program}"

    new_symbols = {}
    exec(python_file_content, new_symbols)

    def assert_is_valid_grammar(maybe_grammar: Any) -> Grammar:
        if (
            not isinstance(maybe_grammar, dict)
            or not all(isinstance(key, str) for key in maybe_grammar)
            or not all(
                isinstance(expansions, list) for expansions in maybe_grammar.values()
            )
            or not all(
                isinstance(expansion, str)
                for expansions in maybe_grammar.values()
                for expansion in expansions
            )
        ):
            print(
                f"avicenna: error: A grammar must be of type "
                + "`Dict[str, List[str]]`.",
                file=stderr,
            )
            sys.exit(GRAMMAR_ERROR)

        return maybe_grammar

    grammar = assert_is_valid_grammar(cast(Grammar, new_symbols["grammar_"]))

    return grammar


def create_parser():
    parser = argparse.ArgumentParser(
        prog="Avicenna",
        description="The avicenna command line interface",
    )

    parser.add_argument(
        "-v", "--version", help="Print the Avicenna version number", action="store_true"
    )

    parser.add_argument(
        "-g",
        "--grammar",
        dest="grammar",
        # type=argparse.FileType("r", encoding="UTF-8"),
        help="""The grammar or input format of the program input. Grammars must declare a rule for a
                non-terminal "<start>" (the start symbol) expanding to a single other non-terminal.
                Python extension files can specify a grammar by declaring a variable `grammar` of type
                `Dict[str, List[str]]`, or (preferably) by specifying a function `grammar()` returning
                Dict objects of that type.""",
    )

    parser.add_argument(
        "-p",
        "--program",
        dest="prop",
        help="The evaluation function, that observes whether the behavior in question occurred."
        " The evaluation function returns a boolean value - True when the behavior is observed, False otherwise",
    )

    parser.add_argument(
        "-i",
        "--inputs",
        dest="initial_inputs",
        help="The initial system inputs that should be used to learn the program's behavior."
        " The initial inputs need to consist of at least one bug-triggering and at least one benign input.",
    )

    parser.add_argument(
        "-l",
        "--log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="set the logging level",
    )

    return parser


def read_files(grammar_file: TextIOWrapper) -> Dict[str, str]:
    return {grammar_file.name: grammar_file.read()}


def ensure_grammar_present(
    stderr, parser: ArgumentParser, args: Namespace, files: Dict[str, str]
) -> None:
    if all(not file.endswith(".py") for file in files):
        parser.print_usage(file=stderr)
        print(
            "avicenna error: You must specify a grammar by `--grammar` "
            "with a file `.py` ending.",
            file=stderr,
        )

        exit(-1)


if __name__ == "__main__":
    exit(main())
