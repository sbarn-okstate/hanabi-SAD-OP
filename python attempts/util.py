# Constants for the maximum number of colors and ranks
kMaxNumColors = 5
kMaxNumRanks = 5


def color_index_to_char(color: int) -> str:
    """Converts a color index into its corresponding character."""
    if 0 <= color < kMaxNumColors:
        return "RYGWB"[color]
    return "X"


def rank_index_to_char(rank: int) -> str:
    """Converts a rank index into its corresponding character."""
    if 0 <= rank < kMaxNumRanks:
        return "12345"[rank]
    return "X"


def parameter_value(params: dict, key: str, default_value):
    """
    Fetches a value associated with a key in the params dictionary
    and converts it to the appropriate type if present. Returns the default value otherwise.
    """
    value = params.get(key, default_value)
    if value == default_value:
        return default_value

    # Type conversion based on the type of the default value
    try:
        if isinstance(default_value, int):
            return int(value)
        elif isinstance(default_value, float):
            return float(value)
        elif isinstance(default_value, bool):
            return value.lower() in ["1", "true", "yes"]
        elif isinstance(default_value, str):
            return str(value)
    except ValueError:
        return default_value

    return default_value


def require(expr: bool):
    """
    Enforces a condition and raises an assertion error if the condition fails.
    In release mode, it would raise a RuntimeError instead of AssertionError.
    """
    assert expr, "Input requirements failed!"
