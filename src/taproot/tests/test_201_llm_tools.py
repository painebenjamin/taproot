from taproot.util import debug_logger

def test_now_tool() -> None:
    from taproot.tasks.generation.text.tools import (
        DateTimeTool,
        DateTimeByTimezoneTool,
        DateTimeByLocationTool,
    )
    with debug_logger() as logger:
        base = DateTimeTool()
        response = base()
        logger.info(f"Base response: {response}")
        assert bool(response), "DateTimeTool failed"
        tz = DateTimeByTimezoneTool()
        response = tz("America/New_York")
        logger.info(f"Timezone response: {response}")
        assert bool(response), "DateTimeByTimezoneTool failed"
        if not DateTimeByLocationTool.is_available():
            logger.warning("Skipping DateTimeByLocationTool test")
            return
        loc = DateTimeByLocationTool()
        response = loc("New York City")
        logger.info(f"Location response: {response}")
        assert bool(response), "DateTimeByLocationTool failed"

def test_weather_tool() -> None:
    from taproot.tasks.generation.text.tools import WeatherAPITool
    with debug_logger() as logger:
        if not WeatherAPITool.is_available():
            logger.warning("Skipping WeatherAPITool test")
            return
        base = WeatherAPITool()
        response = base("New York City")
        logger.info(f"Base response: {response}")
        assert bool(response), "WeatherTool failed"

def test_wikipedia_tool() -> None:
    from taproot.tasks.generation.text.tools import WikipediaTool
    with debug_logger() as logger:
        base = WikipediaTool()
        response = base("New York City")
        logger.info(f"Base response: {response}")
        assert bool(response), "WikipediaTool failed"

def test_fandom_tool() -> None:
    from taproot.tasks.generation.text.tools import FandomTool
    with debug_logger() as logger:
        base = FandomTool()
        response = base("Slay the Spire", "The Silent")
        logger.info(f"Base response: {response}")
        assert bool(response), "FandomTool failed"

def test_duckduckgo_tools() -> None:
    from taproot.tasks.generation.text.tools import (
        DuckDuckGoSearchTool,
        DuckDuckGoSearchReadTool,
        DuckDuckGoNewsTool,
        DuckDuckGoNewsReadTool,
        DuckDuckGoHeadlinesTool
    )
    with debug_logger() as logger:
        # First check headlines as it has no args
        headlines = DuckDuckGoHeadlinesTool()
        response = headlines()
        logger.info(f"Headlines response: {response}")
        assert bool(response), "DuckDuckGoHeadlinesTool failed"
        # Now check all the rest which have one arg
        for tool_cls in [DuckDuckGoSearchTool, DuckDuckGoSearchReadTool, DuckDuckGoNewsTool, DuckDuckGoNewsReadTool]:
            tool = tool_cls()
            response = tool("Slay the Spire")
            logger.info(f"{tool_cls.__name__} response: {response}")
            assert bool(response), f"{tool_cls.__name__} failed"
