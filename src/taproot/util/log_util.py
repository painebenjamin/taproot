"""
Utilities for logging - the global logger, and contexts for debugging.
Copied from fruition.util.log. Thanks, me! :D
"""
from __future__ import annotations

import sys
import http.client

from typing import Any, List, Iterator, Union

from contextlib import contextmanager
from logging import (
    Handler,
    StreamHandler,
    Formatter,
    LogRecord,
    Logger,
    getLogger,
    DEBUG,
)

from .terminal_util import maybe_use_termcolor

__all__ = [
    "logger",
    "debug_logger",
    "ColoredLoggingFormatter",
    "UnifiedLoggingContext",
    "LevelUnifiedLoggingContext",
    "DebugUnifiedLoggingContext",
]

logger = getLogger("taproot")

@contextmanager
def debug_logger(log_level: Union[int, str] = DEBUG) -> Iterator[Logger]:
    """
    A context manager that sets the logger to debug mode for the duration of the context.
    A simple shorthand, useful for testing.
    """
    with LevelUnifiedLoggingContext(log_level):
        yield logger

class ColoredLoggingFormatter(Formatter):
    """
    An extension of the base logging.Formatter that colors the log
    depending on the level.

    This is using termcolor, so it's using terminal color escape sequences.
    These will appear as garbage bytes when not appropriately accounted for.
    """
    def format(self, record: LogRecord) -> str:
        """
        The main ``format`` function enumerates the six possible log levels
        into colors, and formats the log record with that color.

        :param record: The log record to format.
        :returns: The unicode string, colored if the log level is set.
        """
        formatted = super(ColoredLoggingFormatter, self).format(record)
        return {
            "CRITICAL": maybe_use_termcolor(formatted, "red", attrs=["reverse", "blink"]),
            "ERROR": maybe_use_termcolor(formatted, "red"),
            "WARNING": maybe_use_termcolor(formatted, "yellow"),
            "INFO": maybe_use_termcolor(formatted, "green"),
            "DEBUG": maybe_use_termcolor(formatted, "cyan"),
            "NOTSET": formatted,
        }[record.levelname.upper()]

taproot_static_handlers: List[Handler] = []
taproot_static_level: int = 99
taproot_is_frozen: bool = False

class FrozenLogger(Logger):
    """
    A logger that will not allow handlers to be added or removed.
    """
    @classmethod
    def from_logger(cls, logger: Logger) -> FrozenLogger:
        """
        Create a FrozenLogger from a Logger.
        """
        if not isinstance(logger, Logger):
            return logger
        new_logger = cls(logger.name, level=logger.level)
        new_logger.handlers = logger.handlers
        new_logger.propagate = logger.propagate
        new_logger.disabled = logger.disabled
        return new_logger

    def verbose(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log at the verbose level.
        """
        self.log(5, msg, *args, **kwargs)

    def callHandlers(self, record: LogRecord) -> None:
        """
        Pass a record to all relevant handlers.
        This is a copy of the original callHandlers method, but with the
        handler list replaced with the static_handlers list when the logger is frozen.
        """
        global taproot_static_handlers, taproot_is_frozen, taproot_static_level
        from logging import lastResort, raiseExceptions
        c = self
        found = 0
        while c:
            for hdlr in taproot_static_handlers if taproot_is_frozen else c.handlers:
                found = found + 1
                if record.levelno >= taproot_static_level if taproot_is_frozen else hdlr.level:
                    hdlr.handle(record)
            if not c.propagate:
                c = None # type: ignore[assignment]
            else:
                c = c.parent # type: ignore[assignment]
        if (found == 0):
            if lastResort:
                if record.levelno >= lastResort.level:
                    lastResort.handle(record)
            elif raiseExceptions and not self.manager.emittedNoHandlerWarning:
                sys.stderr.write("No handlers could be found for logger"
                                 " \"%s\"\n" % self.name)
                self.manager.emittedNoHandlerWarning = True


class UnifiedLoggingContext:
    """
    A context manager that will remove logger handlers, then set the handler and level for the root
    logger to specified parameters.

    Will set logger variables back to their predefined values on exit.

    :param handler: The handler to set the root logger to.
    :param level: The log level.
    :param silenced: A list of any loggers to silence.
    """

    DEFAULT_FORMAT = (
        "%(asctime)s [%(name)s] %(levelname)s (%(filename)s:%(lineno)s) %(message)s"
    )

    def __init__(self, handler: Handler, level: Union[str, int], silenced: List[str] = []) -> None:
        self.level = level
        self.handler = handler
        self.silenced = silenced

    def __enter__(self) -> None:
        self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def start(self) -> None:
        """
        Find initialized loggers and set their level/handler.
        """
        from logging import _acquireLock, _releaseLock, getLevelName # type: ignore[attr-defined]
        global taproot_static_handlers, taproot_static_level, taproot_is_frozen
        _acquireLock()
        # First freeze future loggers
        taproot_is_frozen = True
        taproot_static_handlers = [self.handler]
        if isinstance(self.level, int):
            taproot_static_level = self.level
        else:
            taproot_static_level = getLevelName(self.level)
        Logger.manager.setLoggerClass(FrozenLogger)

        # Now modify current loggers
        self.handlers = {}
        self.levels = {}
        self.propagates = {}

        self.handlers["root"] = Logger.root.handlers
        self.levels["root"] = Logger.root.level

        Logger.root.handlers = [self.handler]
        Logger.root.setLevel(self.level)

        for loggerName, logger in Logger.manager.loggerDict.items():
            if isinstance(logger, Logger):
                self.handlers[loggerName] = logger.handlers
                self.levels[loggerName] = logger.level
                self.propagates[loggerName] = logger.propagate
                logger.handlers = [self.handler]
                logger.propagate = False
                if loggerName in self.silenced:
                    logger.setLevel(99)
                else:
                    logger.setLevel(self.level)

        def print_http_client(*args: Any, **kwargs: Any) -> None:
            for line in (" ".join(args)).splitlines():
                getLogger("http.client").log(DEBUG, line)

        setattr(http.client, "print", print_http_client)
        http.client.HTTPConnection.debuglevel = 1
        _releaseLock()
        Logger.manager._clear_cache() # type: ignore[attr-defined]

    def stop(self) -> None:
        """
        For loggers that were changed during start(), revert the changes.
        """
        Logger.root.handlers = self.handlers["root"]
        Logger.root.level = self.levels["root"]
        for loggerName, logger in Logger.manager.loggerDict.items():
            if loggerName in self.handlers and loggerName in self.levels and loggerName in self.propagates and isinstance(logger, Logger):
                logger.handlers = self.handlers[loggerName]
                logger.level = self.levels[loggerName]
                logger.propagate = self.propagates[loggerName]
        Logger.manager.setLoggerClass(Logger)

class LevelUnifiedLoggingContext(UnifiedLoggingContext):
    """
    An extension of the UnifiedLoggingContext for use in debugging.

    :param level int: The log level.
    """
    def __init__(self, level: Union[str, int], silenced: List[str] = []) -> None:
        self.level = level
        self.handler = StreamHandler(sys.stdout)
        self.handler.setFormatter(ColoredLoggingFormatter(self.DEFAULT_FORMAT))
        self.silenced = silenced

class DebugUnifiedLoggingContext(LevelUnifiedLoggingContext):
    """
    A shortand for LevelUnifiedLoggingContext(DEBUG)
    """
    def __init__(self, silenced: List[str] = []) -> None:
        super(DebugUnifiedLoggingContext, self).__init__(DEBUG, silenced)
