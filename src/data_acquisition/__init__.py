"""
Data acquisition module for the temperature prediction system.
Includes Wunderground scraper and GEFS fetcher.
"""

from .wunderground_scraper import WundergroundScraper, StationConfig, DailyTemperature

__all__ = ['WundergroundScraper', 'StationConfig', 'DailyTemperature']