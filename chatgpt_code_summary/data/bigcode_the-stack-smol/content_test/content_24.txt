from interpolate import interpolate_doc


foo = """
    hello
    world
"""
bar = "foo bar\nbaz"


class Foo:
    # cf matplotlib's kwdoc.
    __kw__ = "the kw of foo"


@interpolate_doc
def func():
    """
    this is a docstring

    {interpolate_example.foo}

        {bar}

    {Foo!K}
    """


try:
    @interpolate_doc
    def bad_doc():
        """
        fields {must} be preceded by whitespace
        """
except ValueError:
    print("error correctly caught")
