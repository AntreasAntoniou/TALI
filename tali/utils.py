import logging

from rich.logging import RichHandler


def enrichen_logger(logger: logging.Logger):
    ch = RichHandler()

    # create formatter
    formatter = logging.Formatter("%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger
