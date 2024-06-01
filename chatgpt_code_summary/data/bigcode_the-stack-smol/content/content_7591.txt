"""Tests for ArgComb. """

# pylint: disable=unused-argument, unused-variable

from typing import Any, Callable

import pytest

from argcomb import And, Else, InvalidArgumentCombination, Not, Or, Xor, argcomb


def test_default() -> None:
    """Test the ``default`` parameter of :function:``argcomb.__init__``.

    This test also serves to check the basic functionality of
    ``argcomb`` for different types of signature: normal arguments,
    keyword only arguments, positional only arguments, or any
    combination of the above.
    """

    def test_func(
        func: Callable[..., None], kw_only_count: int, pos_only_count: int
    ) -> None:
        """Test a given function ``f``. """

        with pytest.raises(InvalidArgumentCombination):
            func()
        if pos_only_count == 0:
            func(a=1)
            func(a=1, b=1)
            func(a=1, b=None)
            with pytest.raises(InvalidArgumentCombination):
                func(a=None, b=1)
        if kw_only_count < 2:
            func(1)

        if kw_only_count < 2 and pos_only_count < 2:
            func(1, b=1)
            func(1, b=None)
            with pytest.raises(InvalidArgumentCombination):
                func(None, b=1)

        if kw_only_count == 0:
            func(1, 1)
            func(1, None)
            with pytest.raises(InvalidArgumentCombination):
                func(None, 1)

        if pos_only_count < 2:
            with pytest.raises(InvalidArgumentCombination):
                func(b=1)

    @argcomb("a")
    def f(a: Any = None, b: Any = None) -> None:
        ...

    test_func(f, kw_only_count=0, pos_only_count=0)

    @argcomb("a")
    def g(a: Any = None, *, b: Any = None) -> None:
        ...

    test_func(g, kw_only_count=1, pos_only_count=0)

    @argcomb("a")
    def h(*, a: Any = None, b: Any = None) -> None:
        ...

    test_func(h, kw_only_count=2, pos_only_count=0)

    @argcomb("a")
    def i(a: Any = None, /, b: Any = None) -> None:
        ...

    test_func(i, kw_only_count=0, pos_only_count=1)

    @argcomb("a")
    def j(a: Any = None, b: Any = None, /) -> None:
        ...

    test_func(j, kw_only_count=0, pos_only_count=2)

    @argcomb("a")
    def k(a: Any = None, /, *, b: Any = None) -> None:
        ...

    test_func(k, kw_only_count=1, pos_only_count=1)


def test_argument_specs() -> None:
    """Test providing specifications for arguments. """

    @argcomb(a="b", c="d")
    def f(a: Any = None, b: Any = None, c: Any = None, d: Any = None) -> None:
        ...

    # 9 valid combinations
    f()
    f(d=1)
    f(c=1, d=1)
    f(b=1)
    f(b=1, d=1)
    f(b=1, c=1, d=1)
    f(a=1, b=1)
    f(a=1, b=1, d=1)
    f(a=1, b=1, c=1, d=1)

    # 7 invalid combinations
    with pytest.raises(InvalidArgumentCombination):
        f(c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(b=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, d=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, c=1, d=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, b=1, c=1)


def test_value_dependent_specs() -> None:
    """Test specifications which depend on argument value. """

    @argcomb(a={1: "b", 2: "c", 3: "d"})
    def f(a: Any = None, b: Any = None, c: Any = None, d: Any = None) -> None:
        ...

    # valid
    f()
    f(a=1, b=4)
    f(a=2, c=5)
    f(a=3, d=6)
    f(a=1, b=4, c=5)
    f(a=1, b=4, c=5, d=6)
    f(a=1, b=4, d=6)
    f(a=2, c=5, d=6)
    f(a=4)
    f(b=4, c=5)
    f(d=6)

    # invalid
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, c=5)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, c=5, d=6)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=2, b=3)
    with pytest.raises(InvalidArgumentCombination):
        f(a=2, d=4)
    with pytest.raises(InvalidArgumentCombination):
        f(a=3, b=3, c=4)
    with pytest.raises(InvalidArgumentCombination):
        f(a=3)


def test_and() -> None:
    """Test ``And`` condition. """

    @argcomb(And("a", "b"))
    def f(a: Any = None, b: Any = None, c: Any = None) -> None:
        ...

    #  valid
    f(a=1, b=2)
    f(a=1, b=2, c=3)

    # invalid
    with pytest.raises(InvalidArgumentCombination):
        f(a=1)
    with pytest.raises(InvalidArgumentCombination):
        f(b=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, b=None)
    with pytest.raises(InvalidArgumentCombination):
        f(a=None, b=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(b=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(c=1)
    with pytest.raises(InvalidArgumentCombination):
        f()


def test_or() -> None:
    """Test ``Or`` condition. """

    @argcomb(Or("a", "b"))
    def f(a: Any = None, b: Any = None) -> None:
        ...

    # valid
    f(a=1)
    f(b=2)
    f(a=1, b=2)

    # invalid
    with pytest.raises(InvalidArgumentCombination):
        f()


def test_not() -> None:
    """Test ``Not`` condition. """

    @argcomb(Not("a"))
    def f(a: Any = None) -> None:
        ...

    # valid
    f()

    # invalid
    with pytest.raises(InvalidArgumentCombination):
        f(a=1)


def test_xor() -> None:
    """Test ``Xor`` condition. """

    @argcomb(Xor("a", "b", "c"))
    def f(a: Any = None, b: Any = None, c: Any = None) -> None:
        ...

    # valid
    f(a=1)
    f(b=1)
    f(c=1)

    #  invalid
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, b=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(b=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, b=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f()


def test_else() -> None:
    """Test ``Else`` in value dependent specifications. """

    @argcomb(a={1: "b", Else: "c"})
    def f(a: Any = None, b: Any = None, c: Any = None) -> None:
        ...

    # valid
    f(a=2, c=1)

    #  invalid
    with pytest.raises(InvalidArgumentCombination):
        f(a=2, b=1)


def test_nested_condition() -> None:
    """Test a nested condition. """

    @argcomb(Or(And("a", "b"), And("c", "d")))
    def f(a: Any = None, b: Any = None, c: Any = None, d: Any = None) -> None:
        ...

    # valid
    f(a=1, b=1)
    f(c=1, d=1)
    f(a=1, b=1, c=1, d=1)

    # invalid
    with pytest.raises(InvalidArgumentCombination):
        f(a=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f()


def test_argument_named_default() -> None:
    """Test when an argument is named ``default``.

    This collides with a positional only argument named ``default`` in
    the ``argcomb`` signature, but as this is positional only this
    should not matter.
    """

    @argcomb(default="a")
    def f(default: Any = None, a: Any = None) -> None:
        ...

    f(a=1)
    f(default=1, a=1)
    with pytest.raises(InvalidArgumentCombination):
        f(default=1)


def test_arguments_same_name() -> None:
    """Test that a warning is emitted when a function with two
    identically named arguments. """

    @argcomb(a="b")
    def f(a: Any = None, /, b: Any = None, **kwargs: Any) -> None:
        ...

    with pytest.warns(UserWarning):
        f(1, 2, a=3)  # pylint: disable=E1124


def test_default_arguments() -> None:
    """Test that default arguments are correctly recognised when they
    are not ``None``. """

    @argcomb(a="b")
    def f(a: int = 1, b: int = 2) -> None:
        ...

    # valid since ``a`` is the default value
    f(a=1)

    with pytest.raises(InvalidArgumentCombination):
        # invalid since ``b`` is the default value
        f(a=2, b=2)


def test_kwargs() -> None:
    """Test functionality when signature uses ``**kwargs``. """

    @argcomb(a="b")
    def f(**kwargs: Any) -> None:
        ...

    f(a=1, b=1)
    f(b=1, c=1)
    with pytest.raises(InvalidArgumentCombination):
        f(a=1)
