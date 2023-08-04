import logging

LOGGER = logging.getLogger("avicenna")
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s :: %(asctime)s :: %(levelname)-8s :: %(message)s",
)
