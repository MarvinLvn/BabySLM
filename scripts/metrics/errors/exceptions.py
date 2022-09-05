def _print_sublist(entries, num=3):
    """Returns a string containing the `n` first elements of `entries`"""
    if len(entries) <= num:
        return '[' + ', '.join(str(e) for e in entries) + ']'

    return (
        '[' + ', '.join(list(str(e) for e in entries)[:num]) +
        f', ...] and {len(entries) - num} more')

class ValidationError(Exception):
    """Raised when detecting a validation error"""

class FormatError(ValidationError):
    """Raised when detecting a bad format in submission file"""
    def __init__(self, line, message):
        super().__init__(message)
        self._line = line

    def __str__(self):
        return f'bad format (line {self._line}): ' + super().__str__()

class MismatchError(ValidationError):
    """Raised when detecting a mismatch between two sets"""
    def __init__(self, message, expected, observed):
        super().__init__()
        self._message = message

        expected = set(expected)
        observed = set(observed)

        missing = expected - observed
        extra = observed - expected

        if missing or extra:
            self._message += ': '
        if missing:
            self._message += f'missing {_print_sublist(missing)}'
        if missing and extra:
            self._message += ', '
        if extra:
            self._message += f'extra {_print_sublist(extra)}'

    def __str__(self):
        return self._message

