from abc import ABC, abstractmethod
from typing import Any


class SUT(ABC):
    """An abstract system under test class."""

    @abstractmethod
    def process_input(self, inpt: Any) -> Any:
        """
        Process the input to the SUT.

        :param inpt: The input to process.
        :return: The processed input.
        """
        ...

    def gradient_checkpointing(self, enable: bool = False) -> None:
        """
        Toggle gradient checkpointing if implemented.

        :param enable: Whether to enable gradient checkpointing.
        :raises NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("This method is not implemented.")

    @abstractmethod
    def input_valid(self, inpt: Any, cond: Any) -> tuple[bool, Any]:
        """
        Validate input for a condition.

        :param inpt: The input to validate.
        :param cond: The condition to check against.
        :returns: Whether the input is valid and the output of the SUT.
        """
        ...
