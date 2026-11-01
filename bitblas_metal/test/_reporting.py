"""ANSI colors for unittest output, without changing test outcomes."""

from contextlib import contextmanager
import os
import sys
import unittest


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class _ColorStream:
    def __init__(self, stream, enabled):
        self.stream = stream
        self.enabled = enabled
        self.color = GREEN

    def write(self, text):
        if self.enabled and text.strip():
            text = f"{self.color}{text}{RESET}"
        return self.stream.write(text)

    def set_color(self, color):
        self.color = color

    @contextmanager
    def use_color(self, color):
        previous = self.color
        self.color = color
        try:
            yield
        finally:
            self.color = previous

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _ColorTestResult(unittest.TextTestResult):
    def addFailure(self, test, err):
        with self.stream.use_color(RED):
            super().addFailure(test, err)

    def addError(self, test, err):
        with self.stream.use_color(RED):
            super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        with self.stream.use_color(RED if err is not None else GREEN):
            super().addSubTest(test, subtest, err)

    def addSkip(self, test, reason):
        with self.stream.use_color(YELLOW):
            super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        with self.stream.use_color(YELLOW):
            super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        with self.stream.use_color(RED):
            super().addUnexpectedSuccess(test)

    def printErrors(self):
        with self.stream.use_color(RED):
            super().printErrors()
        # TextTestRunner writes its final OK/FAILED summary after printErrors.
        self.stream.set_color(GREEN if self.wasSuccessful() else RED)


class ColorTextTestRunner(unittest.TextTestRunner):
    resultclass = _ColorTestResult

    def __init__(self, *, color="auto", stream=None, **kwargs):
        if color not in ("auto", "always", "never"):
            raise ValueError(f"Unknown color mode: {color!r}.")
        if stream is None:
            stream = sys.stderr
        enabled = color == "always" or (
            color == "auto"
            and "NO_COLOR" not in os.environ
            and getattr(stream, "isatty", lambda: False)()
        )
        super().__init__(stream=_ColorStream(stream, enabled), **kwargs)
