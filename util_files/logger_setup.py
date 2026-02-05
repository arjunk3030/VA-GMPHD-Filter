import logging

# Create a global logger
logger = logging.getLogger("global_logger")
logger.propagate = False

# Only add handler if it doesn't exist yet (avoid duplicates if imported multiple times)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Default level (can be changed externally)
logger.setLevel(logging.INFO)

def set_debug_mode(debug: bool):
    """
    Set logger level based on debug flag.
    """
    logger.setLevel(logging.DEBUG if debug else logging.INFO)