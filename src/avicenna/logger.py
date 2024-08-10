import logging
from typing import Set, Callable, Optional, List

from avicenna.learning.table import Candidate
from avicenna.input.input import Input

LOGGER = logging.getLogger("avicenna")


def configure_logging(level=logging.INFO):
    # Clear root logger handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Clear avicenna logger handlers
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(name)s :: %(asctime)s :: %(levelname)-8s :: %(message)s",
    )


logger = logging.getLogger("avicenna")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s :: %(asctime)s :: %(levelname)-8s :: %(message)s",
)


def generator_report(result: Set[Input]):
    logger.info(f"Generated {len(result)} inputs.")
    logger.debug(f"Generated inputs: {result}")


def runner_report(result: Set[Input]):
    logger.info(f"Executed {len(result)} inputs.")
    logger.debug(f"Executed inputs: {result}")


def learner_report(result: Optional[List[Candidate]]):
    logger.info(f"Learned {len(result)} patterns.")
    logger.debug(f"Learned patterns: {result}")


def relevant_feature_report(features: Optional[Set[str]]):
    logger.info(f"Determined {features} as most relevant features.")


def irrelevant_feature_report(features: Optional[Set[str]]):
    logger.info(f"Excluding {features} from candidate learning.")


def log_execution_with_report(report_func: Optional[Callable]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Executing {func.__name__}")
            try:
                result = func(*args, **kwargs)
                #logger.info(f"{func.__name__} executed successfully")
                report_func(result)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_execution(func, report_func: Optional[Callable] = None):
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}")
        try:
            result = func(*args, **kwargs)
            # logger.info(f"{func.__name__} executed successfully")
            if report_func:
                report_func(result)
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}", exc_info=True)
            raise
    return wrapper

