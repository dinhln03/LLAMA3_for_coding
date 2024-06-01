"""Parser for envpy config parser"""

# Errors

class EnvpyError(Exception):
    """Base class for all envpy errors."""

class MissingConfigError(EnvpyError):
    """Raised when a config item is missing from the environment and has
    no default.
    """

class ValueTypeError(EnvpyError):
    """Raised when a Schema is created with an invalid value type"""

class ParsingError(EnvpyError):
    """Raised when the value pulled from the environment cannot be parsed
    as the given value type."""


# Parsers

def _parse_str(value):
    return value

def _parse_int(value):
    return int(value)

def _parse_float(value):
    return float(value)

def _parse_bool(value):
    is_true = (
        value.upper() == "TRUE"
        or value == "1"
    )
    is_false = (
        value.upper() == "FALSE"
        or value == "0"
    )
    if is_true:
        return True
    elif is_false:
        return False
    else:
        raise ValueError()

PARSERS = {
    str: _parse_str,
    int: _parse_int,
    float: _parse_float,
    bool: _parse_bool,
}


# Parsing logic

SENTINAL = object()

class Schema: #pylint: disable=too-few-public-methods
    """Schema that describes a single environment config item

    Args:
        value_type (optional, default=str): The type that envpy should try to
            parse the environment variable into.
        default (optional): The value that should be used if the variable
            cannot be found in the environment.
    """

    def __init__(self, value_type=str, default=SENTINAL):
        if value_type not in PARSERS:
            raise ValueTypeError()
        self._parser = PARSERS.get(value_type)
        self._default = default

    def parse(self, key, value):
        """Parse the environment value for a given key against the schema.

        Args:
            key: The name of the environment variable.
            value: The value to be parsed.
        """
        if value is not None:
            try:
                return self._parser(value)
            except Exception:
                raise ParsingError("Error parsing {}".format(key))
        elif self._default is not SENTINAL:
            return self._default
        else:
            raise KeyError(key)


def parse_env(config_schema, env):
    """Parse the values from a given environment against a given config schema

    Args:
        config_schema: A dict which maps the variable name to a Schema object
            that describes the requested value.
        env: A dict which represents the value of each variable in the
            environment.
    """
    try:
        return {
            key: item_schema.parse(key, env.get(key))
            for key, item_schema in config_schema.items()
        }
    except KeyError as error:
        raise MissingConfigError(
            "Required config not set: {}".format(error.args[0])
        )
