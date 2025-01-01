import requests
import datetime

from typing import Any, Optional

from ..base import Tool

__all__ = ["WeatherAPITool"]

class WeatherAPITool(Tool):
    """
    A tool that provides weather information using WeatherAPI.
    """
    tool_name = "weather"
    api_key_var: Optional[str] = "WEATHER_API_KEY"

    def get_weather(self, query: str, days_ahead: int=0) -> str:
        """
        Get the weather information for a location.
        """
        response = requests.get(
            "http://api.weatherapi.com/v1/forecast.json",
            params={
                "q": query,
                "key": self.api_key,
                "days": days_ahead + 1,
                "aqi": "yes",
                "alerts": "yes",
            }
        ).json()
        if "error" in response:
            return f"Error {response['error']['code']}: {response['error']['message']}"

        location = response["location"]
        current = response["current"]
        current_aqi = current["air_quality"]["us-epa-index"]

        if current_aqi <= 50:
            aqi_color = "Green"
            aqi_description = "Good"
        elif current_aqi <= 100:
            aqi_color = "Yellow"
            aqi_description = "Moderate"
        elif current_aqi <= 150:
            aqi_color = "Orange"
            aqi_description = "Unhealthy for Sensitive Groups"
        elif current_aqi <= 200:
            aqi_color = "Red"
            aqi_description = "Unhealthy"
        elif current_aqi <= 300:
            aqi_color = "Purple"
            aqi_description = "Very Unhealthy"
        else:
            aqi_color = "Maroon"
            aqi_description = "Hazardous"

        today = response["forecast"]["forecastday"][0]["day"]
        now_dt = datetime.datetime.strptime(location["localtime"], "%Y-%m-%d %H:%M")

        detail_lines = [
            f"Location: {location['name']}",
            f"Local date and time: {now_dt.strftime('%A, %B %d %Y, %I:%m%p')}",
        ]

        for alert in response["alerts"].get("alert", []):
            detail_lines += [
                "",
                f"Alert: {alert['event']}",
                f"Starts: {alert['start']}",
                f"Ends: {alert['end']}",
                f"Severity: {alert['severity']}",
                f"Certainty: {alert['certainty']}",
                f"Urgency: {alert['urgency']}",
                f"Description: {alert['desc']}",
            ]

        detail_lines += [
            "",
            f"Current Condition: {current['condition']['text']}",
            f"Current Temperature: {current['temp_f']}°F/{current['temp_c']}°C",
            f"Current Feels Like: {current['feelslike_f']}°F/{current['feelslike_c']}°C",
            f"Current Wind Speed: {current['wind_mph']} mph/{current['wind_kph']} kph",
            f"Current Gust Speed: {current['gust_mph']} mph/{current['gust_kph']} kph",
            f"Current Humidity: {current['humidity']}%",
            f"Current UV Index: {current['uv']}",
            f"Current Visibility: {current['vis_miles']} miles/{current['vis_km']} km",
            f"Current Air Quality: {current_aqi} ({aqi_color}/{aqi_description})",
        ]

        for i, day in enumerate(response["forecast"]["forecastday"]):
            day_dt = datetime.datetime.strptime(day["date"], "%Y-%m-%d")
            if i == 0:
                day_string = f"Today, {day_dt.strftime('%A, %B %d')}"
            elif i == 1:
                day_string = f"Tomorrow, {day_dt.strftime('%A, %B %d')}"
            else:
                day_string = day_dt.strftime('%A, %B %d %Y')
            day_details = day["day"]
            detail_lines += [
                "",
                f"Forecast for {day_string}:",
                f"Condition: {day_details['condition']['text']}",
                f"Temperature High: {day_details['maxtemp_f']}°F/{day_details['maxtemp_c']}°C",
                f"Temperature Low: {day_details['mintemp_f']}°F/{day_details['mintemp_c']}°C",
                f"Temperature Average: {day_details['avgtemp_f']}°F/{day_details['avgtemp_c']}°C",
                f"Visibility Average: {day_details['avgvis_miles']} miles/{day_details['avgvis_km']} km",
                f"Max Wind Speed: {day_details['maxwind_mph']} mph/{day_details['maxwind_kph']} kph",
                f"Average Humidity: {day_details['avghumidity']}%",
            ]

            if day_details["daily_chance_of_rain"] or day_details["daily_chance_of_snow"]:
                if day_details["daily_chance_of_rain"]:
                    detail_lines.append(f"Chance of Rain: {day_details['daily_chance_of_rain']}%")
                if day_details["daily_chance_of_snow"]:
                    detail_lines.append(f"Chance of Snow: {day_details['daily_chance_of_snow']}%")
                detail_lines.append(f"Total Precipitation: {day_details['totalprecip_in']} inches/{day_details['totalprecip_mm']} mm")
            else:
                detail_lines.append("No precipitation expected.")

        return "\n".join(detail_lines)

    def __call__(
        self,
        location: str,
        days_ahead: int=0,
        **kwargs: Any
    ) -> str:
        """
        Get the weather information for a location, optionally looking some number of days ahead.

        :param location: The location to query, required. Can be as specific as a street address or as generic as a ZIP code or city name.
        :param days_ahead: Optional, the number of days from today to get the weather for. Default to 0 (today). Pass 1 for tomorrow, 2 for the day after tomorrow, etc.
        :return: The description of the weather for the location.
        """
        if not location:
            return "Please provide a location to get the weather for."
        self.cite("https://www.weatherapi.com/", "WeatherAPI")
        return self.get_weather(location, days_ahead=days_ahead)
