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
