from typing import Callable, Dict, Type, Optional
import signal
from avicenna.oracle import OracleResult
from avicenna.input import Input


class ManageTimeout:
    def __init__(self, timeout: int):
        self.timeout = timeout

    def __enter__(self):
        set_alarm(self.timeout)

    def __exit__(self, exc_type, exc_value, traceback):
        cancel_alarm()


class UnexpectedResultError(Exception):
    pass


# Define the handler to be called when the alarm signal is received
def alarm_handler(signum, frame):
    raise TimeoutError("Function call timed out")


# Set the alarm signal handler
signal.signal(signal.SIGALRM, alarm_handler)


def set_alarm(seconds: int):
    signal.alarm(seconds)


def cancel_alarm():
    signal.alarm(0)


def construct_oracle(
    program_under_test: Callable,
    program_oracle: Optional[Callable],
    error_definitions: Dict[Type[Exception], OracleResult],
    timeout: int = 1,
) -> Callable[[Input], OracleResult]:
    if not isinstance(error_definitions, dict):
        raise ValueError(f"Invalid value for expected_error: {error_definitions}")

    if program_oracle:
        return construct_functional_oracle(program_under_test, program_oracle, error_definitions, timeout)

    return construct_failure_oracle(program_under_test, error_definitions, timeout)

def construct_functional_oracle(program_under_test, program_oracle, error_definitions, timeout):
    def oracle(inp: Input) -> OracleResult:
        param = list(map(int, str(inp).strip().split()))  # This might become a problem
        try:
            with ManageTimeout(timeout):
                produced_result = program_under_test(*param)

            expected_result = program_oracle(*param)
            if expected_result != produced_result:
                raise UnexpectedResultError("Results do not match")
        except Exception as e:
            return error_definitions.get(
                type(e), OracleResult.UNDEF
            )  # Default to UNDEF if exception type not found
        return OracleResult.NO_BUG

    return oracle

def construct_failure_oracle(program_under_test, error_definitions, timeout):

    def oracle(inp: Input) -> OracleResult:
        try:
            with ManageTimeout(timeout):
                program_under_test(str(inp))
        except Exception as e:
            return error_definitions.get(
                type(e), OracleResult.UNDEF
            )  # Default to UNDEF if exception type not found
        return OracleResult.NO_BUG

    return oracle
