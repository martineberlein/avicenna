import logging


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


def log_execution(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}", exc_info=True)
            raise
    return wrapper