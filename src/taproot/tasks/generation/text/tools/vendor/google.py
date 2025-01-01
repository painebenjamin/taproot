import time
import requests

from typing import Any, Optional

from ..base import Tool

__all__ = ["GeocodeTool", "TimezoneTool"]

class GoogleAPITool(Tool):
    """
    A base class for google API tools.

    We don't want to introduce dependencies if we don't have to,
    so we use `requests` for this instead of google's official libraries.

    Any tool that doesn't have a 'tool_name' attribute won't be exposed to users.
    Most of the tools here don't, because they're of questionable usefulness and
    only serve to make it more difficult for the LLM to choose the most appropriate one.
    """
    api_key_var: Optional[str] = "GOOGLE_API_KEY"

class GeocodeTool(GoogleAPITool):
    """
    A tool for obtaining latitude and longitude coordinates for an address using the Google Maps API.
    """
    def __call__(self, address: str) -> Any:
        """
        Geocode an address using the Google Maps API. Returns the latitude and longitude coordinates.

        :param address: The address to geocode.
        :return: The geocoded address as a dictionary.
        """
        url = f"https://maps.googleapis.com/maps/api/geocode/json"
        response = requests.get(url, params={"address": address, "key": self.api_key})
        response.raise_for_status()
        return response.json()

class TimezoneTool(GoogleAPITool):
    """
    A tool for obtaining the timezone from a location using the Google Maps API.
    """
    def __call__(
        self,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Any:
        """
        Get the timezone for a location using the Google Maps API.

        :param location: The location to get the timezone for - can be as specific as an address or generic as a country. If provided, the latitude and longitude are ignored.
        :param latitude: The latitude of the location.
        :param longitude: The longitude of the location.
        :return: The timezone of the location as a string.
        """
        if location is not None:
            geocode = GeocodeTool()
            response = geocode(location)
            latitude = response["results"][0]["geometry"]["location"]["lat"]
            longitude = response["results"][0]["geometry"]["location"]["lng"]
        elif latitude is None or longitude is None:
            raise ValueError("Either 'location' or both 'latitude' and 'longitude' must be provided.")

        timestamp = time.time()
        url = f"https://maps.googleapis.com/maps/api/timezone/json"
        response = requests.get(
            url,
            params={
                "key": self.api_key,
                "location": f"{latitude},{longitude}",
                "timestamp": timestamp
            }
        )
        response.raise_for_status()
        return response.json()
