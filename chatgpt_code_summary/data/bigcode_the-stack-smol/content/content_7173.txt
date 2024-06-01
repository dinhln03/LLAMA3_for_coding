from ..options import Option, OptionHandler


class LocaleOptionsMixin(OptionHandler):
    """
    Mixin which adds a locale option to option handlers.
    """
    # The locale to use
    locale = Option(
        help="the locale to use for parsing the numbers",
        default="en_US"
    )
