from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Optional, TypeVar

from goodboy.errors import Error
from goodboy.messages import DEFAULT_MESSAGES, MessageCollectionType, type_name
from goodboy.schema import Rule, SchemaWithUtils

N = TypeVar("N")


class NumericBase(Generic[N], SchemaWithUtils):
    """
    Abstract base class for Int/Float schemas, should not be used directly. Use
    :class:`Int` or :class:`Float` instead.
    """

    def __init__(
        self,
        *,
        allow_none: bool = False,
        messages: MessageCollectionType = DEFAULT_MESSAGES,
        rules: list[Rule] = [],
        less_than: Optional[N] = None,
        less_or_equal_to: Optional[N] = None,
        greater_than: Optional[N] = None,
        greater_or_equal_to: Optional[N] = None,
        allowed: Optional[list[N]] = None,
    ):
        super().__init__(allow_none=allow_none, messages=messages, rules=rules)
        self._less_than = less_than
        self._less_or_equal_to = less_or_equal_to
        self._greater_than = greater_than
        self._greater_or_equal_to = greater_or_equal_to
        self._allowed = allowed

    def _validate(
        self, value: Any, typecast: bool, context: dict[str, Any] = {}
    ) -> tuple[Optional[N], list[Error]]:
        value, type_errors = self._validate_exact_type(value)

        if type_errors:
            return None, type_errors

        errors = []

        if self._allowed is not None and value not in self._allowed:
            errors.append(self._error("not_allowed", {"allowed": self._allowed}))

        if self._less_than is not None and value >= self._less_than:
            errors.append(
                self._error("greater_or_equal_to", {"value": self._less_than})
            )

        if self._less_or_equal_to is not None and value > self._less_or_equal_to:
            errors.append(
                self._error("greater_than", {"value": self._less_or_equal_to})
            )

        if self._greater_than is not None and value <= self._greater_than:
            errors.append(
                self._error("less_or_equal_to", {"value": self._greater_than})
            )

        if self._greater_or_equal_to is not None and value < self._greater_or_equal_to:
            errors.append(
                self._error("less_than", {"value": self._greater_or_equal_to})
            )

        value, rule_errors = self._call_rules(value, typecast, context)

        return value, errors + rule_errors

    @abstractmethod
    def _validate_exact_type(self, value: Any) -> tuple[Optional[N], list[Error]]:
        ...


class Float(NumericBase[float]):
    """
    Accept ``float`` values. Integer values are converted to floats.

    When type casting enabled, strings and other values with magic method
    `__float__ <https://docs.python.org/3/reference/datamodel.html#object.__float__>`_
    are converted to floats.

    :param allow_none: If true, value is allowed to be ``None``.
    :param messages: Override error messages.
    :param rules: Custom validation rules.
    :param less_than: Accept only values less than option value.
    :param less_or_equal_to: Accept only values less than or equal to option value.
    :param greater_than: Accept only values greater than option value.
    :param greater_or_equal_to: Accept only values greater than or equal to option
        value.
    :param allowed: Allow only certain values.
    """

    def _typecast(
        self, input: Any, context: dict[str, Any] = {}
    ) -> tuple[Optional[float], list[Error]]:
        if isinstance(input, float):
            return input, []

        if isinstance(input, int):
            return float(input), []

        if not isinstance(input, str):
            return None, [
                self._error("unexpected_type", {"expected_type": type_name("float")})
            ]

        try:
            return float(input), []
        except ValueError:
            return None, [self._error("invalid_numeric_format")]

    def _validate_exact_type(self, value: Any) -> tuple[Optional[float], list[Error]]:
        if isinstance(value, float):
            return value, []
        elif isinstance(value, int):
            return float(value), []
        else:
            return None, [
                self._error("unexpected_type", {"expected_type": type_name("float")})
            ]


class Int(NumericBase[int]):
    """
    Accept ``int`` values.

    When type casting enabled, strings and other values with magic method
    `__int__ <https://docs.python.org/3/reference/datamodel.html#object.__int__>`_ are
    converted to integers.

    :param allow_none: If true, value is allowed to be ``None``.
    :param messages: Override error messages.
    :param rules: Custom validation rules.
    :param less_than: Accept only values less than option value.
    :param less_or_equal_to: Accept only values less than or equal to option value.
    :param greater_than: Accept only values greater than option value.
    :param greater_or_equal_to: Accept only values greater than or equal to option
        value.
    :param allowed: Allow only certain values.
    """

    def _typecast(
        self, input: Any, context: dict[str, Any] = {}
    ) -> tuple[Optional[int], list[Error]]:
        if isinstance(input, int):
            return input, []

        if not isinstance(input, str):
            return None, [
                self._error("unexpected_type", {"expected_type": type_name("int")})
            ]

        try:
            return int(input), []
        except ValueError:
            return None, [self._error("invalid_integer_format")]

    def _validate_exact_type(self, value: Any) -> tuple[Optional[int], list[Error]]:
        if not isinstance(value, int):
            return None, [
                self._error("unexpected_type", {"expected_type": type_name("int")})
            ]
        else:
            return value, []
