import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("global_logger")
