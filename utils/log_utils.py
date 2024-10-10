""" Logging setup and utils """
import logging
import sys

LOG_LEVEL = logging.getLevelName("INFO")
formatter = logging.Formatter(
    "%(asctime)s | %(name)s - %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)


def console_handler():
    result = logging.StreamHandler(sys.stdout)
    result.setFormatter(formatter)
    result.setLevel(LOG_LEVEL)
    return result


def file_handler(logging_file):
    result = logging.FileHandler(logging_file)
    result.setFormatter(formatter)
    result.setLevel(LOG_LEVEL)
    return result


def logger(logger_name, logging_file=None):
    result = logging.getLogger(name=logger_name)
    result.addHandler(console_handler())
    if logging_file is not None:
        result.addHandler(file_handler(logging_file))
    result.setLevel(LOG_LEVEL)

    return result
