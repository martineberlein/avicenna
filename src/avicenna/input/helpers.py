from debugging_framework.input.oracle import OracleResult


def map_to_bool(result: OracleResult) -> bool:
    match result:
        case OracleResult.FAILING:
            return True
        case OracleResult.PASSING:
            return False
        case _:
            return False
