# import moonshine as ms
# from moonshine.curves import discount_factor


from .curves import get_discount_factor
from .instruments import price_cashflow


def egg(num_eggs: int) -> None:
    """prints the number of eggs.

    Arguments:
        num_eggs {int} -- The number of eggs

    Returns:
        None.
    """
    print(f"We have {num_eggs} eggs")


def main() -> None:

    discount_factor = get_discount_factor(0.02, 20)
    price = price_cashflow(10, 0.02, 10)

    bumped_price = 2.0 + price
    bumped_price += price + 3

    egg(123)

    print(price)
    print(discount_factor)
    print(bumped_price)


if __name__ == "__main__":
    main()
