from typing import Callable, Dict, Type
import signal
from avicenna.oracle import OracleResult
from avicenna.input import Input


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
    program_oracle,
    program_under_test,
    error_definitions: Dict[Type[Exception], OracleResult],
    timeout: int = 1,
) -> Callable[[Input], OracleResult]:
    if not isinstance(error_definitions, dict):
        raise ValueError(f"Invalid value for expected_error: {error_definitions}")

    def oracle(inp: Input) -> OracleResult:
        param = list(map(int, str(inp).strip().split()))
        try:
            expected_result = program_oracle(*param)
            set_alarm(timeout)  # Set an alarm
            produced_result = program_under_test(*param)
            cancel_alarm()  # Cancel the alarm
            if expected_result != produced_result:
                raise UnexpectedResultError("Results do not match")
        except Exception as e:
            return error_definitions.get(
                type(e), OracleResult.UNDEF
            )  # Default to UNDEF if exception type not found
        return OracleResult.NO_BUG

    return oracle
