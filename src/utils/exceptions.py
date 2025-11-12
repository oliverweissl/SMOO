class SMOOException(Exception):
    """A general exception for the SMOO Framework."""

    pass


class ExceededIterationBudget(SMOOException):
    """Raised when an iteration exceeds the budget."""

    pass
