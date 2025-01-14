"""
Utilities for logging - the global logger, and contexts for debugging.
Adapted from fruition.util.log. Thanks, me! :D
"""
from __future__ import annotations

import os
import sys
import warnings
import http.client

from typing import Any, List, Iterator, Union, Optional, Dict

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
    with LevelUnifiedLoggingContext(level=log_level):
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

    Update from fruition: This class now also leverages the environment to try and configure remote
    logging services, when relevant packages are installed and environment variables are set. The
    supported services are:

    - Loggly: set LOGGLY_TOKEN, and optionally LOGGLY_TAG and LOGGLY_URL. Must have loggly-python-handler[http-transport] installed.

    :param handler: The handler to set the root logger to.
    :param level: The log level.
    :param silenced: A list of any loggers to silence.
    """
    DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s (%(filename)s:%(lineno)s) %(message)s"
    LOGGLY_FORMAT = '{ "loggerName":"%(name)s", "timestamp":"%(asctime)s", "pathName":"%(pathname)s", "logRecordCreationTime":"%(created)f", "functionName":"%(funcName)s", "levelNo":"%(levelno)s", "lineNo":"%(lineno)d", "time":"%(msecs)d", "levelName":"%(levelname)s", "message":"%(message)s" }'

    stored_handlers: Dict[str, List[Handler]]
    stored_levels: Dict[str, int]
    stored_propagates: Dict[str, bool]

    def __init__(
        self,
        handler: Optional[Handler] = None,
        level: Union[str, int] = DEBUG,
        include_http: bool = False,
        silenced: List[str] = []
    ) -> None:
        self.level = level
        self.include_http = include_http
        self.handler = handler
        self.silenced = silenced
        self.stored_handlers = {}
        self.stored_levels = {}
        self.stored_propagates = {}

    def silence_urllib3_connectionpool(self) -> None:
        """
        Silences the urllib3.connectionpool logger.

        This is necessary when using HTTP-based loggers, if you wish to
        avoid infinitely recursing logs of sending logs of sending logs...
        """
        if "urllib3.connectionpool" not in self.silenced:
            self.silenced.append("urllib3.connectionpool")

    @property
    def handlers(self) -> List[Handler]:
        """
        All configured handlers.
        """
        handlers = [self.handler, self.loggly_handler, self.fruition_handler]
        return [h for h in handlers if h is not None]

    @property
    def loggly_handler(self) -> Optional[Handler]:
        """
        A loggly handler, if the environment is configured for it.
        """
        if not hasattr(self, "_loggly_handler"):
            self._loggly_handler = self.get_loggly_handler()
            if self._loggly_handler is not None:
                self.silence_urllib3_connectionpool()
        return self._loggly_handler

    @property
    def fruition_handler(self) -> Optional[Handler]:
        """
        A fruition handler, if the environment is configured for it.
        """
        if not hasattr(self, "_fruition_handler"):
            self._fruition_handler = self.get_fruition_handler()
            if self._fruition_handler is not None:
                self.silence_urllib3_connectionpool()
        return self._fruition_handler

    def get_loggly_handler(self) -> Optional[Handler]:
        """
        A loggly handler, if the environment is configured for it.
        """
        token = os.getenv("LOGGLY_TOKEN", None)
        if token is not None:
            url = os.getenv("LOGGLY_URL", "http://logs-01.loggly.com/inputs")
            tag = os.getenv("LOGGLY_TAG", "http")
            full_url = "/".join([url.strip("/"), token, "tag", tag])
            try:
                from loggly.handlers import HTTPSHandler # type: ignore[import-untyped,import-not-found,unused-ignore]
                handler = HTTPSHandler(full_url)
                handler.setFormatter(Formatter(self.LOGGLY_FORMAT))
                return handler # type: ignore[no-any-return]
            except ImportError:
                warnings.warn("Loggly handler requested, but loggly-python-handler not installed.")
        return None

    def get_fruition_handler(self) -> Optional[Handler]:
        """
        A fruition handler, if the environment is configured for it.
        """
        url = os.getenv("FRUITION_LOG_URL", None)
        if url is not None:
            tag = os.getenv("FRUITION_LOG_TAG", "taproot")
            authentication = os.getenv("FRUITION_LOG_AUTH", None)
            interval = float(os.getenv("FRUITION_LOG_INTERVAL", 5))
            kwargs: Dict[str, Any] = {
                "interval": interval,
                "tag": tag
            }

            if authentication is not None:
                authentication_parts = authentication.split(":")
                assert len(authentication_parts) >= 2, "Authentication should of of the form method:username_or_token{:password}"
                method = authentication_parts[0]
                username_or_token = authentication_parts[1]
                password = authentication_parts[2] if len(authentication_parts) > 2 else None
                kwargs["authentication"] = method
                kwargs["username"] = username_or_token
                kwargs["password"] = password

            try:
                from fruition.ext.log.handler import LogAggregateHandler # type: ignore[import-not-found,unused-ignore]
                handler = LogAggregateHandler(url, **kwargs)
                handler.setFormatter(Formatter(self.DEFAULT_FORMAT))
                return handler
            except ImportError:
                warnings.warn("Fruition handler requested, but fruition not installed.")
        return None

    def __enter__(self) -> None:
        self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def start(self) -> None:
        """
        Find initialized loggers and set their level/handler.
        """
        from logging import getLevelName
        try:
            from logging import _acquireLock, _releaseLock # type: ignore[attr-defined]
        except ImportError:
            _acquireLock = lambda: None
            _releaseLock = lambda: None
        global taproot_static_handlers, taproot_static_level, taproot_is_frozen
        _acquireLock()
        # First freeze future loggers
        taproot_is_frozen = True
        taproot_static_handlers = self.handlers
        if isinstance(self.level, int):
            taproot_static_level = self.level
        else:
            taproot_static_level = getLevelName(self.level)

        Logger.manager.setLoggerClass(FrozenLogger)

        # Now modify current loggers
        self.stored_handlers.clear()
        self.stored_levels.clear()
        self.stored_propagates.clear()

        self.stored_handlers["root"] = Logger.root.handlers
        self.stored_levels["root"] = Logger.root.level

        Logger.root.handlers = self.handlers
        Logger.root.setLevel(self.level)

        for loggerName, logger in Logger.manager.loggerDict.items():
            if isinstance(logger, Logger):
                self.stored_handlers[loggerName] = logger.handlers
                self.stored_levels[loggerName] = logger.level
                self.stored_propagates[loggerName] = logger.propagate

                logger.handlers = self.handlers
                logger.propagate = False
                if loggerName in self.silenced:
                    logger.setLevel(99)
                else:
                    logger.setLevel(self.level)

        def print_http_client(*args: Any, **kwargs: Any) -> None:
            for line in (" ".join(args)).splitlines():
                getLogger("http.client").log(DEBUG, line)

        if self.include_http:
            setattr(http.client, "print", print_http_client)
            http.client.HTTPConnection.debuglevel = 1

        _releaseLock()
        Logger.manager._clear_cache() # type: ignore[attr-defined]

    def stop(self) -> None:
        """
        For loggers that were changed during start(), revert the changes.
        """
        Logger.root.handlers = self.stored_handlers["root"]
        Logger.root.level = self.stored_levels["root"]
        for loggerName, logger in Logger.manager.loggerDict.items():
            if loggerName in self.stored_handlers and loggerName in self.stored_levels and loggerName in self.stored_propagates and isinstance(logger, Logger):
                logger.handlers = self.stored_handlers[loggerName]
                logger.level = self.stored_levels[loggerName]
                logger.propagate = self.stored_propagates[loggerName]
        Logger.manager.setLoggerClass(Logger)

class LevelUnifiedLoggingContext(UnifiedLoggingContext):
    """
    An extension of the UnifiedLoggingContext for use in debugging.

    :param level int: The log level.
    """
    def __init__(
        self,
        handler: Optional[Handler] = None,
        level: Union[str, int] = DEBUG,
        include_http: bool = False,
        silenced: List[str] = []
    ) -> None:
        super(LevelUnifiedLoggingContext, self).__init__(
            handler=StreamHandler(sys.stdout),
            level=level,
            include_http=include_http,
            silenced=silenced
        )
        self.handler.setFormatter( # type: ignore[union-attr]
            ColoredLoggingFormatter(self.DEFAULT_FORMAT)
        )

class DebugUnifiedLoggingContext(LevelUnifiedLoggingContext):
    """
    A shortand for LevelUnifiedLoggingContext(DEBUG)
    """
    def __init__(
        self,
        handler: Optional[Handler] = None,
        level: Union[str, int] = DEBUG,
        include_http: bool = False,
        silenced: List[str] = []
    ) -> None:
        super(DebugUnifiedLoggingContext, self).__init__(
            level=DEBUG,
            silenced=silenced
        )
