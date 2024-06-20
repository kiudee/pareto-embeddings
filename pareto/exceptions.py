__all__ = ["ParetoException", "NoJobException", "MissingGPUException"]


class ParetoException(Exception):
    """Base class for other exceptions"""

    pass


class NoJobException(ParetoException):
    """Raised when the runner was unable to fetch a job."""

    pass


class MissingGPUException(ParetoException):
    """Raised when a job requires a GPU, but none is found."""

    pass
