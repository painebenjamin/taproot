import re
import datetime

from difflib import get_close_matches
from typing import Optional

from .base import Tool

__all__ = [
    "DateTimeTool",
    "DateTimeByTimezoneTool",
    "DateTimeByLocationTool",
]

def pytz_is_available() -> bool:
    """
    Checks if the pytz library is available

    :return: True if pytz is available, False otherwise.
    """
    try:
        import pytz
        return True
    except ImportError:
        return False

class DateTimeTool(Tool):
    """
    Gets the current time and date.
    """
    def get_timezone_by_name_or_id(self, name: str) -> datetime.tzinfo:
        """
        Given a string timezone name, returns the closest valid timezone object based on string edit distance.
        
        :param name: The timezone name to match.
        :return: The closest matching timezone object.
        """
        import pytz
        # Get the list of all valid timezone names
        all_timezones = dict([
            (tz.upper(), tz)
            for tz in pytz.all_timezones
        ])

        # Find the closest matching timezone name
        closest_match = get_close_matches(name.upper(), list(all_timezones.keys()), n=1)
        if not closest_match:
            raise ValueError(f"No matching timezone found for '{name}'")

        # Return the timezone object for the closest match
        return pytz.timezone(all_timezones[closest_match[0]])

    def get_timezone_by_location(self, location: str) -> datetime.tzinfo:
        """
        Given a location, returns the timezone object for that location.

        Looks up the latitude and longitude of the location using Google's geocoding API, then uses Google's timezone API
        to retrieve the timezone for that location. Finally, uses pytz to get the timezone object.
        """
        import pytz
        from .vendor.google import TimezoneTool
        timezone_tool = TimezoneTool()
        timezone = timezone_tool(location=location)
        return pytz.timezone(timezone["timeZoneId"])

    def format_timezone(self, timezone: datetime.tzinfo) -> str:
        """
        Formats a timezone object into a human-readable string.
        :param timezone: The timezone object to format.
        :return: The human-readable timezone name.
        """
        # Get the current time in the timezone
        now = datetime.datetime.now(timezone)
        
        # Calculate the UTC offset
        offset_seconds = now.utcoffset().total_seconds() # type: ignore[union-attr]
        hours, remainder = divmod(abs(offset_seconds), 3600)
        minutes = remainder // 60
        sign = "-" if offset_seconds < 0 else "+"

        # Format the UTC offset as HH:MM
        offset_formatted = f"{sign}{int(hours):02}:{int(minutes):02}"

        # Return the formatted string
        try:
            tzname = now.tzname()
        except:
            tzname = None
        if tzname and not re.search(r"^[\d:.\-]+$", tzname):
            return f"{timezone.zone}, {offset_formatted} ({tzname})" # type: ignore[attr-defined]
        return f"{timezone.zone}, {offset_formatted} ({timezone._tzname})" # type: ignore[attr-defined]

    def __call__(
        self,
        time_zone: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        """ 
        Gets the current time and date.

        :param time_zone: The time zone name or ID to use. Optional.
        :param location: The location to get the time for, in lieu of providing a time zone. Optional.
        :return: The current time and date.
        """
        note: Optional[str] = None
        tz: Optional[datetime.tzinfo] = None

        if not pytz_is_available():
            note = "pytz is not installed, letting system default as configured by the operating system."
        elif location:
            try:
                tz = self.get_timezone_by_location(location)
            except ValueError:
                note = "could not find a timezone for the location '{location}', defaulting to UTC."
        elif time_zone:
            try:
                tz = self.get_timezone_by_name_or_id(time_zone)
            except ValueError:
                note = "could not find a timezone by the name '{time_zone}', defaulting to UTC."

        # Get the current time and date
        now = datetime.datetime.now(tz)

        # Return the current time and date in nice format
        dtm = now.strftime("%Y-%m-%d %I:%M:%S %p")
        message = f"The current time and date is {dtm}"

        if tz is not None:
            message = f"{message} in {self.format_timezone(tz)}."
        else:
            message = f"{message}."

        if note:
            message = f"{message} Note: {note}"

        return message

class DateTimeByTimezoneTool(DateTimeTool):
    """
    Gets the current time and date for a given timezone.
    """
    tool_name = "datetime-by-timezone"

    def __call__(self, time_zone: str) -> str: # type: ignore[override]
        """
        Gets the current time and date for a given timezone.

        :param time_zone: The time zone ID, like 'America/New_York' or 'UTC'.
        :return: The current time and date.
        """
        return super().__call__(time_zone=time_zone)

class DateTimeByLocationTool(DateTimeTool):
    """
    Gets the current time and date for a given location.
    """
    tool_name = "datetime-by-location"

    @classmethod
    def is_available(cls) -> bool:
        """
        Checks if the tool is available.
        """
        from .vendor.google import TimezoneTool
        return super().is_available() and TimezoneTool.is_available()

    def __call__(self, location: str) -> str: # type: ignore[override]
        """
        Gets the current time and date for a given location.

        :param location: The location to get the time for. Can be an address, city, state or country.
        :return: The current time and date.
        """
        return super().__call__(location=location)
